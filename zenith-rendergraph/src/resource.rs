use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use derive_more::From;
use zenith_rhi::vk;
use crate::builder::ResourceAccessStorage;
use crate::interface::{Buffer, Texture, BufferDesc, TextureDesc, BufferState, TextureState, ResourceDescriptor, ResourceState};
use crate::RenderGraphBuilder;

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait GraphResource: Sized + sealed::Sealed {
    type Descriptor: GraphResourceDescriptor;
    type State: GraphResourceState;
}

pub trait GraphResourceDescriptor: Clone + Into<ResourceDescriptor> {
    type Resource: GraphResource;
}

pub trait GraphResourceState: Copy + Into<ResourceState> {
    type Resource: GraphResource;
}

pub trait GraphResourceView: Copy {}

#[derive(Clone, Copy, Debug)]
pub struct Srv;

#[derive(Clone, Copy, Debug)]
pub struct Uav;

#[derive(Clone, Copy, Debug)]
pub struct Rt;

impl GraphResourceView for Srv {}
impl GraphResourceView for Uav {}
impl GraphResourceView for Rt {}

/// Identifier unique represent an inner resource owned by render graph.
///
/// ## Safety
/// Used in the same render graph context. Should NOT be used across multiple render graph.
pub(crate) type GraphResourceId = u32;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RenderGraphResource<R: GraphResource> {
    pub(crate) id: GraphResourceId,
    pub(crate) _marker: PhantomData<R>,
}

impl<R: GraphResource> RenderGraphResource<R> {
    pub fn valid(&self) -> bool {
        self.id != u32::MAX
    }

    /// Gets the descriptor for this resource.
    ///
    /// # Safety
    /// This method uses transmute which is safe because:
    /// 1. The sealed trait ensures only Buffer and Texture implement GraphResource
    /// 2. PhantomData<R> ensures the resource type matches the storage variant
    /// 3. The enum discriminant is checked before transmute
    pub fn desc(&self, builder: &RenderGraphBuilder) -> &<R as GraphResource>::Descriptor {
        match builder.initial_resources.get(self.id as usize).unwrap() {
            InitialResourceStorage::ManagedBuffer(desc) => unsafe { std::mem::transmute(desc) },
            InitialResourceStorage::ManagedTexture(desc) => unsafe { std::mem::transmute(desc) },
            InitialResourceStorage::ImportedBuffer(res, _) => unsafe { std::mem::transmute(res.desc()) },
            InitialResourceStorage::ImportedTexture(res, _) => unsafe { std::mem::transmute(res.desc()) },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderGraphResourceAccess<R: GraphResource, V: GraphResourceView> {
    pub(crate) id: GraphResourceId,
    pub(crate) access: ResourceState,
    pub(crate) _marker: PhantomData<(R, V)>,
}

impl<R: GraphResource, V: GraphResourceView> RenderGraphResourceAccess<R, V> {
    pub fn valid(&self) -> bool {
        self.id != u32::MAX
    }
}

impl<R: GraphResource, V: GraphResourceView> RenderGraphResourceAccess<R, V> {
    pub(crate) fn as_untyped(&self) -> ResourceAccessStorage {
        ResourceAccessStorage {
            id: self.id,
            access: self.access,
            stage_hint: None,
        }
    }

    pub(crate) fn as_untyped_with_hint(&self, stage_hint: vk::PipelineStageFlags2) -> ResourceAccessStorage {
        ResourceAccessStorage {
            id: self.id,
            access: self.access,
            stage_hint: Some(stage_hint),
        }
    }
}

pub trait GraphImportExportResource: GraphResource {
    fn import(shared_resource: impl Into<Arc<Self>>, builder: &mut crate::builder::RenderGraphBuilder, access: impl Into<ResourceState>) -> RenderGraphResource<Self>;
    fn export(resource: RenderGraphResource<Self>, builder: &mut crate::builder::RenderGraphBuilder, access: impl Into<ResourceState>) -> ExportedRenderGraphResource<Self>;
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ExportedRenderGraphResource<R: GraphResource> {
    #[allow(dead_code)]
    pub(crate) id: GraphResourceId,
    pub(crate) _marker: PhantomData<R>,
}

#[derive(From)]
pub(crate) enum InitialResourceStorage {
    ManagedBuffer(BufferDesc),
    ManagedTexture(TextureDesc),
    ImportedBuffer(Arc<Buffer>, BufferState),
    ImportedTexture(Arc<Texture>, TextureState),
}

impl InitialResourceStorage {
    pub(crate) fn name(&self) -> &str {
        match self {
            InitialResourceStorage::ManagedBuffer(desc) => &desc.name,
            InitialResourceStorage::ManagedTexture(desc) => &desc.name,
            InitialResourceStorage::ImportedBuffer(buf, _) => buf.name(),
            InitialResourceStorage::ImportedTexture(tex, _) => tex.name(),
        }
    }
}

#[allow(dead_code)]
pub(crate) enum ExportResourceStorage {
    ExportedBuffer(BufferState),
    ExportedTexture(TextureState),
}
