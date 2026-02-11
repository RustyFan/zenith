use derive_more::{From, TryInto};
use std::marker::PhantomData;
use std::sync::Arc;
use crate::builder::{RenderGraphBuilder};
use crate::resource::{sealed, ExportedRenderGraphResource, GraphImportExportResource, GraphResource, GraphResourceDescriptor, GraphResourceState, RenderGraphResource};

macro_rules! render_graph_resource_interface {
	($($res:ident => $res_ty:ty, $res_desc:ident => $res_desc_ty:ty, $res_state:ident => $res_state_ty:ty),+) => {
        $(
            pub(crate) type $res = $res_ty;
            pub(crate) type $res_desc = $res_desc_ty;
            pub(crate) type $res_state = $res_state_ty;

            impl sealed::Sealed for $res {}

            impl GraphResource for $res_ty {
                type Descriptor = $res_desc;
                type State = $res_state_ty;
            }

            impl GraphResourceDescriptor for $res_desc_ty {
                type Resource = $res;
            }

            impl GraphResourceState for $res_state_ty {
                type Resource = $res;
            }

            impl GraphImportExportResource for $res_ty {
                fn import(
                    shared_resource: impl Into<Arc<$res_ty>>,
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
        )+

        #[derive(From)]
        pub enum ResourceDescriptor {
            $(
                $res(<$res as GraphResource>::Descriptor),
            )+
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, From, TryInto)]
        pub enum ResourceState {
            $(
                $res($res_state),
            )+
        }
	};
}

render_graph_resource_interface!(
    Buffer => zenith_rhi::Buffer, BufferDesc => zenith_rhi::BufferDesc, BufferState => zenith_rhi::BufferState,
    Texture => zenith_rhi::Texture, TextureDesc => zenith_rhi::TextureDesc, TextureState => zenith_rhi::TextureState
);
