//! Canonical vocabulary of the thin-film stack model.
//!
//! Discriminants mirror `navette.structure.types` exactly (pinned by test):
//! the solver reads raw ints, so these values are wire format, not style.

use std::fmt;

/// Statistical law used when drawing fabrication errors.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorType {
  Gaussian = 0,
  Uniform = 1,
  Combined = 2,
}

/// Per-interface roughness form factor (solver contract, sigma in nm).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoughnessType {
  None = 0,
  Linear = 1,
  Step = 2,
  Exponential = 3,
  Gaussian = 4,
  NevotCroce = 5,
}

/// Slots of the per-layer error-application vector.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorMask {
  Thickness = 0,
  NReal = 1,
  NImag = 2,
  Roughness = 3,
  InhDelta = 4,
  Interface = 5,
}

/// Slots of the per-layer status mask (`Layer::mask`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerMask {
  Active = 0,
  Coherent = 1,
  Inhomogen = 2,
  Roughness = 3,
}

/// Slots of the per-group optimization mask.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptMask {
  Thickness = 0,
  N = 1,
  K = 2,
  Roughness = 3,
  InhDelta = 4,
  Interface = 5,
  Material = 6,
}

/// Design role of a layer (markers delimit stacks).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
  Ambient = 0,
  Film = 1,
  Substrate = 2,
}

/// Composition role of an architect block (declared, never inferred).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
  Stack = 0,
  Films = 1,
}

macro_rules! impl_int_coercion {
  ($t:ty, [$($v:ident),+]) => {
    impl $t {
      /// Integer value on the wire (solver contract).
      pub fn as_i32(self) -> i32 {
        self as i32
      }
      /// Fail-closed coercion: unknown ints are errors, never defaults.
      pub fn try_from_i32(v: i32) -> Result<Self, String> {
        match v {
          $(x if x == Self::$v as i32 => Ok(Self::$v),)+
          _ => Err(format!(
            "{}: invalid discriminant {} (valid: {}).",
            stringify!($t),
            v,
            [$(Self::$v as i32),+]
              .iter()
              .map(|n| n.to_string())
              .collect::<Vec<_>>()
              .join(", ")
          )),
        }
      }
    }
    impl fmt::Display for $t {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({})", self, *self as i32)
      }
    }
  };
}

impl_int_coercion!(ErrorType, [Gaussian, Uniform, Combined]);
impl_int_coercion!(RoughnessType, [None, Linear, Step, Exponential, Gaussian, NevotCroce]);
impl_int_coercion!(ErrorMask, [Thickness, NReal, NImag, Roughness, InhDelta, Interface]);
impl_int_coercion!(LayerMask, [Active, Coherent, Inhomogen, Roughness]);
impl_int_coercion!(OptMask, [Thickness, N, K, Roughness, InhDelta, Interface, Material]);
impl_int_coercion!(LayerType, [Ambient, Film, Substrate]);
impl_int_coercion!(BlockKind, [Stack, Films]);

#[cfg(test)]
mod tests {
  use super::*;

  /// Oracle twin: discriminants must equal the Python IntEnums bit-for-bit.
  /// (Mirrors validation/regression: fingerprint pins these on both sides.)
  #[test]
  fn discriminants_match_python() {
    assert_eq!(ErrorType::Gaussian.as_i32(), 0);
    assert_eq!(ErrorType::Uniform.as_i32(), 1);
    assert_eq!(ErrorType::Combined.as_i32(), 2);
    let rough = [0, 1, 2, 3, 4, 5];
    for (i, v) in [
      RoughnessType::None,
      RoughnessType::Linear,
      RoughnessType::Step,
      RoughnessType::Exponential,
      RoughnessType::Gaussian,
      RoughnessType::NevotCroce,
    ]
    .iter()
    .enumerate()
    {
      assert_eq!(v.as_i32(), rough[i]);
    }
    assert_eq!(
      [
        ErrorMask::Thickness,
        ErrorMask::NReal,
        ErrorMask::NImag,
        ErrorMask::Roughness,
        ErrorMask::InhDelta,
        ErrorMask::Interface,
      ]
      .iter()
      .map(|m| m.as_i32())
      .collect::<Vec<_>>(),
      [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
      [LayerMask::Active, LayerMask::Coherent, LayerMask::Inhomogen, LayerMask::Roughness]
        .iter()
        .map(|m| m.as_i32())
        .collect::<Vec<_>>(),
      [0, 1, 2, 3]
    );
    assert_eq!(OptMask::Thickness.as_i32(), 0);
    assert_eq!(OptMask::Material.as_i32(), 6);
    assert_eq!(LayerType::Ambient.as_i32(), 0);
    assert_eq!(LayerType::Film.as_i32(), 1);
    assert_eq!(LayerType::Substrate.as_i32(), 2);
    assert_eq!(BlockKind::Stack.as_i32(), 0);
    assert_eq!(BlockKind::Films.as_i32(), 1);
  }

  #[test]
  fn int_coercion_is_fail_closed() {
    assert_eq!(RoughnessType::try_from_i32(5), Ok(RoughnessType::NevotCroce));
    assert_eq!(LayerType::try_from_i32(1), Ok(LayerType::Film));
    assert!(RoughnessType::try_from_i32(6).is_err());
    assert!(RoughnessType::try_from_i32(-1).is_err());
    assert!(LayerType::try_from_i32(7).is_err());
    assert!(BlockKind::try_from_i32(2).is_err());
    assert!(OptMask::try_from_i32(7).is_err());
  }
}
