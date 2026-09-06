//! State-schema version gate.
//!
//! Mirrors `navette.structure.types.SCHEMA_VERSION`: v1 is the baseline,
//! there is no past (untagged states are malformed, not legacy).

/// Current state-schema version (structure states + config states).
pub const SCHEMA_VERSION: u32 = 1;

/// Refuse states not written at the current schema version.
pub fn check_schema_version(found: Option<u32>, what: &str) -> Result<(), String> {
  match found {
    None => Err(format!("{what}: missing schema_version tag (malformed state).")),
    Some(v) if v == SCHEMA_VERSION => Ok(()),
    Some(v) => Err(format!(
      "{what}: schema_version {v} unsupported \
       (this code reads {SCHEMA_VERSION}); refusing a stale state."
    )),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn version_gate() {
    assert_eq!(SCHEMA_VERSION, 1);
    assert!(check_schema_version(Some(1), "Layer").is_ok());
    assert!(check_schema_version(None, "Layer").is_err());
    assert!(check_schema_version(Some(0), "Layer").is_err());
    assert!(check_schema_version(Some(1000), "Layer").is_err());
  }
}
