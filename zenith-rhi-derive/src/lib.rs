#![allow(non_snake_case)]

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, spanned::Spanned, Data, DeriveInput, Fields, Lit, Meta, PathArguments, Type,
    TypeArray, TypePath,
};

#[proc_macro_attribute]
#[allow(non_snake_case)]
pub fn DeviceObject(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    let ident = input.ident.clone();
    let generics = input.generics.clone();
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut output_struct = input.clone();

    let syn::Fields::Named(fields_named) = &mut output_struct.fields else {
        return syn::Error::new(
            output_struct.span(),
            "DeviceObject only supports structs with named fields",
        )
        .to_compile_error()
        .into();
    };

    // Disallow existing `device` field to avoid ambiguity.
    if fields_named.named.iter().any(|f| f.ident.as_ref().is_some_and(|id| id == "device")) {
        return syn::Error::new(
            fields_named.span(),
            "DeviceObject: struct already has a `device` field; remove it and let the macro inject `pub(crate) device: ash::Device`",
        )
        .to_compile_error()
        .into();
    }

    // Inject: `pub(crate) device: ::ash::Device`
    let device_field: syn::Field = syn::parse_quote! {
        pub(crate) device: ::ash::Device
    };
    fields_named.named.push(device_field);

    // Generate sealed + DeviceObject impls (crate-local paths).
    let expanded = quote! {
        #output_struct

        impl #impl_generics crate::device::sealed::Sealed for #ident #ty_generics #where_clause {}

        impl #impl_generics crate::device::DeviceObject for #ident #ty_generics #where_clause {
            #[inline]
            fn device(&self) -> &::ash::Device { &self.device }

            #[inline]
            fn set_device(&mut self, device: ::ash::Device) { self.device = device; }
        }
    };

    expanded.into()
}

#[proc_macro_derive(VertexLayout)]
pub fn derive_vertex_layout(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let ident = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Require repr(C) for deterministic field layout.
    if !has_repr_c(&input.attrs) {
        return syn::Error::new(
            input.span(),
            "VertexLayout requires #[repr(C)] on the vertex struct to ensure stable field offsets",
        )
        .to_compile_error()
        .into();
    }

    let fields = match input.data {
        Data::Struct(s) => match s.fields {
            Fields::Named(named) => named.named,
            Fields::Unnamed(_) | Fields::Unit => {
                return syn::Error::new(
                    ident.span(),
                    "VertexLayout only supports structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new(ident.span(), "VertexLayout only supports structs")
                .to_compile_error()
                .into();
        }
    };

    let mut attr_inits = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let field_ident = match &field.ident {
            Some(id) => id,
            None => {
                return syn::Error::new(field.span(), "expected named field")
                    .to_compile_error()
                    .into();
            }
        };

        let fmt = match vk_format_for_type(&field.ty) {
            Ok(ts) => ts,
            Err(e) => return e.to_compile_error().into(),
        };

        let location = i as u32;
        let init = quote! {
            ::zenith_rhi::VertexAttribute {
                location: #location,
                binding: 0u32,
                format: #fmt,
                offset: ::zenith_rhi::memoffset::offset_of!(Self, #field_ident) as u32,
            }
        };
        attr_inits.push(init);
    }

    let expanded = quote! {
        impl #impl_generics ::zenith_rhi::VertexLayout for #ident #ty_generics #where_clause {
            fn vertex_layout() -> (::zenith_rhi::VertexBinding, ::std::vec::Vec<::zenith_rhi::VertexAttribute>) {
                let binding = ::zenith_rhi::VertexBinding {
                    binding: 0u32,
                    stride: ::core::mem::size_of::<Self>() as u32,
                    input_rate: ::zenith_rhi::vk::VertexInputRate::VERTEX,
                };
                let attributes = ::std::vec![#(#attr_inits),*];
                (binding, attributes)
            }
        }
    };

    expanded.into()
}

fn has_repr_c(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }
        let Ok(meta) = attr.parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated) else {
            continue;
        };
        for m in meta {
            match m {
                Meta::Path(p) if p.is_ident("C") => return true,
                _ => {}
            }
        }
    }
    false
}

fn vk_format_for_type(ty: &Type) -> Result<proc_macro2::TokenStream, syn::Error> {
    // Arrays like [f32; 3]
    if let Type::Array(TypeArray { elem, len, .. }) = ty {
        let n = match len {
            syn::Expr::Lit(expr_lit) => match &expr_lit.lit {
                Lit::Int(li) => li.base10_parse::<usize>().ok(),
                _ => None,
            },
            _ => None,
        }
        .ok_or_else(|| syn::Error::new(ty.span(), "array length must be an integer literal"))?;

        let elem = elem.as_ref();
        return vk_format_for_scalar_array(elem, n);
    }

    // Scalars like f32 / u32 / i32
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(ident) = path.get_ident() {
            return match ident.to_string().as_str() {
                "f32" => Ok(quote!(::zenith_rhi::vk::Format::R32_SFLOAT)),
                "u32" => Ok(quote!(::zenith_rhi::vk::Format::R32_UINT)),
                "i32" => Ok(quote!(::zenith_rhi::vk::Format::R32_SINT)),
                _ => Err(syn::Error::new(
                    ty.span(),
                    format!("unsupported vertex field type `{}`", ident),
                )),
            };
        }

        // Reject generic wrappers for now (Vec, etc.)
        if let Some(seg) = path.segments.last() {
            if matches!(seg.arguments, PathArguments::AngleBracketed(_)) {
                return Err(syn::Error::new(
                    ty.span(),
                    "unsupported generic vertex field type; use primitives or arrays like [f32; N]",
                ));
            }
        }
    }

    Err(syn::Error::new(
        ty.span(),
        format!("unsupported vertex field type `{}`", ty.to_token_stream()),
    ))
}

fn vk_format_for_scalar_array(elem: &Type, n: usize) -> Result<proc_macro2::TokenStream, syn::Error> {
    let scalar = if let Type::Path(TypePath { path, .. }) = elem {
        path.get_ident().map(|i| i.to_string())
    } else {
        None
    }
    .ok_or_else(|| syn::Error::new(elem.span(), "array element type must be a primitive"))?;

    match (scalar.as_str(), n) {
        ("f32", 2) => Ok(quote!(::zenith_rhi::vk::Format::R32G32_SFLOAT)),
        ("f32", 3) => Ok(quote!(::zenith_rhi::vk::Format::R32G32B32_SFLOAT)),
        ("f32", 4) => Ok(quote!(::zenith_rhi::vk::Format::R32G32B32A32_SFLOAT)),

        ("u32", 2) => Ok(quote!(::zenith_rhi::vk::Format::R32G32_UINT)),
        ("u32", 3) => Ok(quote!(::zenith_rhi::vk::Format::R32G32B32_UINT)),
        ("u32", 4) => Ok(quote!(::zenith_rhi::vk::Format::R32G32B32A32_UINT)),

        ("i32", 2) => Ok(quote!(::zenith_rhi::vk::Format::R32G32_SINT)),
        ("i32", 3) => Ok(quote!(::zenith_rhi::vk::Format::R32G32B32_SINT)),
        ("i32", 4) => Ok(quote!(::zenith_rhi::vk::Format::R32G32B32A32_SINT)),

        _ => Err(syn::Error::new(
            elem.span(),
            format!("unsupported vertex array type `[{scalar}; {n}]` (supported: f32/u32/i32 with N=2..4)"),
        )),
    }
}


