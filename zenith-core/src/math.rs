use std::cmp::Ordering;
use std::f32::consts::{FRAC_1_PI, PI};
use derive_more::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, From, Into, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use glam::FloatExt;

#[derive(Deref, DerefMut, From, Into, Default, Debug, Clone, Copy, PartialEq, PartialOrd, Neg, Add, Sub, Mul, Div, Rem, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign)]
pub struct Degree(f32);

impl PartialEq<f32> for Degree {
    fn eq(&self, other: &f32) -> bool {
        self.0.eq(other)
    }
}

impl PartialOrd<f32> for Degree {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl Degree {
    #[inline]
    pub fn new(val: f32) -> Self {
        Self(val)
    }

    #[inline]
    pub fn to_radians(self) -> Radians {
        self.into()
    }

    #[inline]
    pub fn clamp(&self, min: Degree, max: Degree) -> Degree {
        self.0.clamp(min.0, max.0).into()
    }

    #[inline]
    pub fn lerp(&self, rhs: Degree, factor: f32) -> Degree {
        self.0.lerp(rhs.0, factor).into()
    }
}

#[derive(Deref, DerefMut, From, Into, Default, Debug, Clone, Copy, PartialEq, PartialOrd, Neg, Add, Sub, Mul, Div, Rem, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign)]
pub struct Radians(f32);

impl PartialEq<f32> for Radians {
    fn eq(&self, other: &f32) -> bool {
        self.0.eq(other)
    }
}

impl PartialOrd<f32> for Radians {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl Radians {
    #[inline]
    pub fn new(val: f32) -> Self {
        Self(val)
    }

    #[inline]
    pub fn to_degree(self) -> Degree {
        self.into()
    }

    #[inline]
    pub fn clamp(&self, min: Radians, max: Radians) -> Radians {
        self.0.clamp(min.0, max.0).into()
    }

    #[inline]
    pub fn lerp(&self, rhs: Radians, factor: f32) -> Radians {
        self.0.lerp(rhs.0, factor).into()
    }
}

impl From<Degree> for Radians {
    fn from(value: Degree) -> Self {
        Self(value.0 / 180.0 * PI)
    }
}

impl From<Radians> for Degree {
    fn from(value: Radians) -> Self {
        Self(value.0 * FRAC_1_PI * 180.0)
    }
}