use derive_more::{From, TryInto};
use std::marker::PhantomData;
use std::sync::Arc;
use crate::builder::{RenderGraphBuilder};
use crate::graph::ResourceStorage;
use crate::resource::{sealed, ExportedRenderGraphResource, GraphImportExportResource, GraphResource, GraphResourceDescriptor, GraphResourceState, RenderGraphResource};

// macro_rules! render_graph_resource_interface {
// 	($($res:ident => $res_ty:ty, $res_desc:ident => $res_desc_ty:ty, $res_state:ident => $res_state_ty:ty),+) => {
//         $(
//             // pub type $res = $res_ty;
//             // pub type $res_desc = $res_desc_ty;
//             // pub type $res_state = $res_state_ty;
//
//             impl GraphResource for $res_ty {
//                 type Descriptor = $res_desc;
//             }
//
//             impl GraphResourceDescriptor for $res_desc_ty {
//                 type Resource = $res;
//             }
//
//             impl GraphResourceState for $res_state_ty {
//                 type Resource = $res;
//             }
//
//             impl GraphImportExportResource for $res_ty {
//                 fn import(shared_resource: impl Into<RenderResource<Self>>, name: &str, builder: &mut RenderGraphBuilder, access: impl Into<GraphResourceAccess>) -> RenderGraphResource<Self> {
//                     let id = builder.initial_resources.len() as u32;
//                     let uses = access.into().try_into().expect("Inconsistent import resource access!");
//                     builder.initial_resources.push((name.to_owned(), shared_resource.into(), uses).into());
//
//                     RenderGraphResource {
//                         id,
//                         _marker: PhantomData,
//                     }
//                 }
//
//                 fn export(_resource: RenderGraphResource<Self>, _builder: &mut RenderGraphBuilder, _access: impl Into<GraphResourceAccess>) -> ExportedRenderGraphResource<Self> {
//                     unimplemented!()
//                 }
//             }
//         )+
//
//         #[derive(From)]
//         pub enum ResourceDescriptor {
//             $(
//                 $res(<$res as GraphResource>::Descriptor),
//             )+
//         }
//
//         #[derive(Debug, Clone, Copy, PartialEq, Eq, From, TryInto)]
//         pub enum GraphResourceAccess {
//             $(
//                 $res($res_state),
//             )+
//         }
// 	};
// }

pub(crate) type Buffer = zenith_rhi::Buffer;
pub(crate) type BufferDesc = zenith_rhi::BufferDesc;
pub(crate) type BufferState = zenith_rhi::BufferState;

pub(crate) type Texture = zenith_rhi::Texture;
pub(crate) type TextureDesc = zenith_rhi::TextureDesc;
pub(crate) type TextureState = zenith_rhi::TextureState;

impl sealed::Sealed for Buffer {}

impl GraphResource for Buffer {
    type Descriptor = BufferDesc;
    type State = BufferState;

    #[doc(hidden)]
    fn from_storage(storage: &ResourceStorage) -> &Self {
        storage.as_buffer()
    }
}

impl GraphResourceDescriptor for BufferDesc {
    type Resource = Buffer;
}

impl GraphResourceState for BufferState {
    type Resource = Buffer;
}

impl GraphImportExportResource for Buffer {
    fn import(
        shared_resource: impl Into<Arc<Buffer>>,
        builder: &mut RenderGraphBuilder,
        access: impl Into<ResourceState>
    ) -> RenderGraphResource<Self> {
        let id = builder.initial_resources.len() as u32;
        let uses = access.into().try_into().expect("Inconsistent import resource access!");
        builder.initial_resources.push((shared_resource.into(), uses).into());

        RenderGraphResource {
            id,
            _marker: PhantomData,
        }
    }

    fn export(_resource: RenderGraphResource<Self>, _builder: &mut RenderGraphBuilder, _access: impl Into<ResourceState>) -> ExportedRenderGraphResource<Self> {
        unimplemented!()
    }
}

impl sealed::Sealed for Texture {}

impl GraphResource for Texture {
    type Descriptor = TextureDesc;
    type State = TextureState;

    #[doc(hidden)]
    fn from_storage(storage: &ResourceStorage) -> &Self {
        storage.as_texture()
    }
}

impl GraphResourceDescriptor for TextureDesc {
    type Resource = Texture;
}

impl GraphResourceState for TextureState {
    type Resource = Texture;
}

impl GraphImportExportResource for Texture {
    fn import(
        shared_resource: impl Into<Arc<Self>>,
        builder: &mut RenderGraphBuilder,
        access: impl Into<ResourceState>
    ) -> RenderGraphResource<Self> {
        let id = builder.initial_resources.len() as u32;
        let state = access.into().try_into().expect("Inconsistent import resource access!");
        builder.initial_resources.push((shared_resource.into(), state).into());

        RenderGraphResource {
            id,
            _marker: PhantomData,
        }
    }

    fn export(_resource: RenderGraphResource<Self>, _builder: &mut RenderGraphBuilder, _access: impl Into<ResourceState>) -> ExportedRenderGraphResource<Self> {
        unimplemented!()
    }
}

#[derive(From)]
pub enum ResourceDescriptor {
    Buffer(BufferDesc),
    Texture(TextureDesc),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, From, TryInto)]
pub enum ResourceState {
    Buffer(BufferState),
    Texture(TextureState),
}

// render_graph_resource_interface!(
//     Buffer => Arc<zenith_rhi::Buffer>, BufferDesc => zenith_rhi::BufferDesc, BufferState => zenith_rhi::BufferState,
//     Texture => Arc<zenith_rhi::Texture>, TextureDesc => zenith_rhi::TextureDesc, TextureState => zenith_rhi::TextureState
// );

// #[derive(Deref, DerefMut, From, Clone, Debug)]
// pub struct RenderResource<T: GraphResource>(T);
//
// impl<T: GraphResource> RenderResource<T> {
//     pub fn new(resource: T) -> Self {
//         Self(resource)
//     }
// }