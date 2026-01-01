use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use derive_more::From;
use crate::builder::ResourceAccessStorage;

use crate::interface::{Buffer, Texture, BufferDesc, TextureDesc, BufferState, TextureState, ResourceDescriptor, ResourceState};

pub trait GraphResource {
    type Descriptor: GraphResourceDescriptor;
}

pub trait GraphResourceDescriptor: Clone + Into<ResourceDescriptor> {
    type Resource: GraphResource;
}

pub trait GraphResourceState: Copy + Eq {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderGraphResourceAccess<R: GraphResource, V: GraphResourceView> {
    pub(crate) id: GraphResourceId,
    pub(crate) access: ResourceState,
    pub(crate) _marker: PhantomData<(R, V)>,
}

impl<R: GraphResource, V: GraphResourceView> RenderGraphResourceAccess<R, V> {
    pub(crate) fn as_untyped(&self) -> ResourceAccessStorage {
        ResourceAccessStorage {
            id: self.id,
            access: self.access,
        }
    }
}

pub trait GraphImportExportResource: GraphResource + Sized {
    fn import(shared_resource: impl Into<Arc<Self>>, name: &str, builder: &mut crate::builder::RenderGraphBuilder, access: impl Into<ResourceState>) -> RenderGraphResource<Self>;
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
    ManagedBuffer(String, BufferDesc),
    ManagedTexture(String, TextureDesc),
    ImportedBuffer(String, Arc<Buffer>, BufferState),
    ImportedTexture(String, Arc<Texture>, TextureState),
}

impl InitialResourceStorage {
    pub(crate) fn name(&self) -> &str {
        match self {
            InitialResourceStorage::ManagedBuffer(name, _) => name,
            InitialResourceStorage::ManagedTexture(name, _) => name,
            InitialResourceStorage::ImportedBuffer(name, _, _) => name,
            InitialResourceStorage::ImportedTexture(name, _, _) => name,
        }
    }
}

#[allow(dead_code)]
pub(crate) enum ExportResourceStorage {
    ExportedBuffer(BufferState),
    ExportedTexture(TextureState),
}
