//! Parser for the CIE DataCite JSON table envelope.
//!
//! Every reference file under `navette/data/CIE/` shares one envelope: a
//! top-level object with a `data` mapping of column-name →
//! `{values: [...], …metadata}`. Per-column siblings (`title`, `unit`,
//! `quantity`, `wavelength_first/last/step`, `description`) are metadata
//! and ignored — only `values` is read. The parser is fully generic: **no
//! column name is hardcoded**.
//!
//! In-tree census (2026-09-06, 97 files — the cases this must survive):
//! - `cmf/`: `lambda` + `x_bar/y_bar/z_bar(lambda)` (1931, 471 rows),
//!   `xbar_10/…` (1964, 471), `xbar_F/…` (cfb 2°, 441),
//!   `xbar_F,10/…` (cfb 10° — note the comma).
//! - `cc/`: `x/y/z(lambda)` (1931), `x_10/y_10/z_10(lambda)` (1964),
//!   `l_MB/m_MB/s_MB(lambda)` (smb MacLeod-Boynton, 89 rows — deliberately
//!   NOT an XYZ triplet).
//! - `lef/`: `V(lambda)`, `V_10(lambda)`, `V'(lambda)` (prime!),
//!   `V_mes;m(lambda)` (semicolon!), plus the lambda-free
//!   `CIE_max_sle_mesopic.json` (`m` + `K_m,mes;m`, 11 rows).
//! - `sds/`: `lambda` + one bare-name column (`FL1`, `FL3.10` with dots,
//!   `LED-B1`/`LED-RGB1` with dashes, `HP1`, `S_A(lambda)`,
//!   `S_D65(lambda)`, `S_ID65(lambda)`, `CIE illuminant C` with spaces).
//! - root: `lambda` + `S_L41(lambda)`.
//! - All values numeric (int or float), all columns equal length per file.
//! - Duplicate column names keep the last occurrence (JSON object
//!   semantics — identical to Python's `json.load`, so twins agree).

/// One parsed CIE table: named float columns.
#[derive(Clone, Debug)]
pub struct CieTable {
  columns: Vec<(String, Vec<f64>)>,
}

/// Parse CIE DataCite JSON text into a [`CieTable`].
///
/// Refuses (with the culprit named): invalid JSON, missing/empty `data`
/// mapping, columns without a `values` array, empty or non-numeric or
/// non-finite values, and columns of unequal length.
pub fn parse_cie_tables(text: &str) -> Result<CieTable, String> {
  let root: serde_json::Value =
    serde_json::from_str(text).map_err(|e| format!("CIE table: invalid JSON: {e}"))?;
  let data = root
    .get("data")
    .and_then(|v| v.as_object())
    .ok_or_else(|| {
      "CIE table: missing 'data' mapping (not a CIE table document).".to_string()
    })?;
  if data.is_empty() {
    return Err("CIE table: 'data' mapping is empty.".to_string());
  }
  let mut columns = Vec::with_capacity(data.len());
  for (name, entry) in data.iter() {
    let values = entry
      .get("values")
      .and_then(|v| v.as_array())
      .ok_or_else(|| format!("CIE table: column '{name}' has no 'values' array."))?;
    if values.is_empty() {
      return Err(format!("CIE table: column '{name}' has no rows."));
    }
    let mut col = Vec::with_capacity(values.len());
    for (i, v) in values.iter().enumerate() {
      let x = v
        .as_f64()
        .ok_or_else(|| format!("CIE table: column '{name}' row {i} is not a number."))?;
      if !x.is_finite() {
        return Err(format!("CIE table: column '{name}' row {i} is not finite."));
      }
      col.push(x);
    }
    columns.push((name.clone(), col));
  }
  let n0 = columns[0].1.len();
  for (name, col) in &columns {
    if col.len() != n0 {
      return Err(format!(
        "CIE table: column '{name}' has {} rows, expected {n0}.",
        col.len()
      ));
    }
  }
  Ok(CieTable { columns })
}

impl CieTable {
  /// Column names (serde key order — compare as sets, not lists).
  pub fn column_names(&self) -> Vec<&str> {
    self.columns.iter().map(|(n, _)| n.as_str()).collect()
  }

  /// One column by exact name.
  pub fn column(&self, name: &str) -> Option<&[f64]> {
    self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c.as_slice())
  }

  /// The `lambda` wavelength axis, if the file carries one (`None` for
  /// lambda-free tables such as the mesopic `K_m` adaptation table).
  pub fn lambda(&self) -> Option<&[f64]> {
    self.column("lambda")
  }

  /// XYZ-ish column names in x,y,z order, found by remainder-grouping.
  ///
  /// A candidate stem is a `(lambda)`-suffixed column whose name, lowercased
  /// with `_`, `,`, whitespace stripped, starts with `x`/`y`/`z`; stems
  /// group by remainder (`bar` → 1931, `bar10` → 1964, `barf`/`barf10` →
  /// Stockman-Sharpe, `` → cc 1931, `10` → cc 1964). Exactly one full
  /// `{x,y,z}` group must exist. MacLeod-Boynton (`l/m/s_MB`), `V'`,
  /// `S_D65` and bare SDS names never qualify (wrong first letter or no
  /// `(lambda)` suffix — the suffix rule is what keeps
  /// `CIE illuminant C`-style columns out).
  pub fn xyz_column_names(&self) -> Result<[String; 3], String> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, BTreeMap<char, String>> = BTreeMap::new();
    for (name, _) in &self.columns {
      let Some((axis, rest)) = xyz_stem(name) else { continue };
      groups.entry(rest).or_default().insert(axis, name.clone());
    }
    let full: Vec<&BTreeMap<char, String>> = groups
      .values()
      .filter(|m| {
        m.len() == 3 && m.contains_key(&'x') && m.contains_key(&'y') && m.contains_key(&'z')
      })
      .collect();
    match full.as_slice() {
      [m] => Ok([m[&'x'].clone(), m[&'y'].clone(), m[&'z'].clone()]),
      [] => Err(format!(
        "CIE table: no XYZ triplet found (columns: {}).",
        self.column_names().join(", ")
      )),
      _ => {
        let mut rests: Vec<&str> =
          groups.keys().map(|r| if r.is_empty() { "(bare)" } else { r }).collect();
        rests.sort_unstable();
        Err(format!(
          "CIE table: ambiguous XYZ triplets ({}) — qualify explicitly.",
          rests.join(", ")
        ))
      }
    }
  }
}

/// `(axis, remainder)` for triplet candidates; `None` for anything else.
fn xyz_stem(name: &str) -> Option<(char, String)> {
  let base = name.strip_suffix("(lambda)")?;
  let norm: String = base
    .to_lowercase()
    .chars()
    .filter(|c| !matches!(c, '_' | ',' | ' ' | '\t'))
    .collect();
  let mut chars = norm.chars();
  let axis = chars.next()?;
  if !matches!(axis, 'x' | 'y' | 'z') {
    return None;
  }
  Some((axis, chars.collect()))
}

// ---------------------------------------------------------------------------
// Tests (inline fixtures — the core does no I/O, so each file-family
// convention is replicated as a literal, plus the error paths).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn doc(data: &str) -> String {
    format!("{{\"data\": {{{data}}}}}")
  }

  fn col(name: &str, vals: &str) -> String {
    format!("\"{name}\": {{\"values\": [{vals}], \"unit\": \"nm\"}}")
  }

  #[test]
  fn cmf_1931_convention_triplet_in_order() {
    let t = parse_cie_tables(&doc(&[
      col("lambda", "360, 361"),
      col("z_bar(lambda)", "0.0006, 0.0007"),
      col("x_bar(lambda)", "0.0001, 0.0002"),
      col("y_bar(lambda)", "0.0000, 0.0000"),
    ]
    .join(",")))
    .unwrap();
    assert_eq!(t.lambda().unwrap(), &[360.0, 361.0]);
    assert_eq!(
      t.xyz_column_names().unwrap(),
      [
        "x_bar(lambda)".to_string(),
        "y_bar(lambda)".to_string(),
        "z_bar(lambda)".to_string()
      ]
    );
  }

  #[test]
  fn all_xyz_conventions_resolve() {
    // (x-names, y-names, z-names) per family convention.
    let families = [
      (["x_bar(lambda)", "y_bar(lambda)", "z_bar(lambda)"]),
      (["xbar_10(lambda)", "ybar_10(lambda)", "zbar_10(lambda)"]),
      (["xbar_F(lambda)", "ybar_F(lambda)", "zbar_F(lambda)"]),
      (["xbar_F,10(lambda)", "ybar_F,10(lambda)", "zbar_F,10(lambda)"]),
      (["x(lambda)", "y(lambda)", "z(lambda)"]),
      (["x_10(lambda)", "y_10(lambda)", "z_10(lambda)"]),
    ];
    for names in families {
      let cols: Vec<String> = std::iter::once(col("lambda", "1, 2"))
        .chain(names.iter().map(|n| col(n, "0.1, 0.2")))
        .collect();
      let t = parse_cie_tables(&doc(&cols.join(","))).unwrap();
      let got = t.xyz_column_names().unwrap();
      assert_eq!(got, names.map(str::to_string), "convention {names:?}");
    }
  }

  #[test]
  fn macleod_boynton_is_no_triplet() {
    let t = parse_cie_tables(&doc(&[
      col("lambda", "390, 395"),
      col("l_MB(lambda)", "0.69, 0.70"),
      col("m_MB(lambda)", "0.31, 0.30"),
      col("s_MB(lambda)", "0.85, 0.84"),
    ]
    .join(",")))
    .unwrap();
    let err = t.xyz_column_names().unwrap_err();
    assert!(err.contains("no XYZ triplet"), "{err}");
  }

  #[test]
  fn sds_bare_names_parse_triplet_refused() {
    let t = parse_cie_tables(&doc(&[
      col("lambda", "380, 381"),
      col("FL1", "1.0, 2.0"),
    ]
    .join(",")))
    .unwrap();
    assert_eq!(t.column_names(), ["FL1", "lambda"]);
    assert_eq!(t.column("FL1").unwrap(), &[1.0, 2.0]);
    let err = t.xyz_column_names().unwrap_err();
    assert!(err.contains("FL1"), "{err}");
  }

  #[test]
  fn lambda_free_table_parses() {
    let t = parse_cie_tables(&doc(&[
      col("m", "0, 1"),
      col("K_m,mes;m", "1700.13, 683"),
    ]
    .join(",")))
    .unwrap();
    assert!(t.lambda().is_none());
    assert_eq!(t.column("K_m,mes;m").unwrap(), &[1700.13, 683.0]);
  }

  #[test]
  fn ambiguous_triplet_refused() {
    let t = parse_cie_tables(&doc(&[
      col("lambda", "1, 2"),
      col("x(lambda)", "0.1, 0.2"),
      col("y(lambda)", "0.1, 0.2"),
      col("z(lambda)", "0.1, 0.2"),
      col("x_bar(lambda)", "0.1, 0.2"),
      col("y_bar(lambda)", "0.1, 0.2"),
      col("z_bar(lambda)", "0.1, 0.2"),
    ]
    .join(",")))
    .unwrap();
    let err = t.xyz_column_names().unwrap_err();
    assert!(err.contains("ambiguous"), "{err}");
  }

  #[test]
  fn error_paths_name_the_culprit() {
    assert!(parse_cie_tables("not json").unwrap_err().contains("invalid JSON"));
    assert!(parse_cie_tables("{}").unwrap_err().contains("missing 'data'"));
    assert!(parse_cie_tables("{\"data\": {}}").unwrap_err().contains("empty"));
    let no_vals = doc(&"\"a\": {\"unit\": \"nm\"}".to_string());
    assert!(parse_cie_tables(&no_vals).unwrap_err().contains("'a'"));
    let empty = doc(&col("a", ""));
    assert!(parse_cie_tables(&empty).unwrap_err().contains("'a'"));
    let bad = doc(&[col("a", "1, 2"), col("b", "1, \"x\"")].join(","));
    let err = parse_cie_tables(&bad).unwrap_err();
    assert!(err.contains("'b'") && err.contains("row 1"), "{err}");
    let uneven = doc(&[col("a", "1, 2"), col("b", "1, 2, 3")].join(","));
    let err = parse_cie_tables(&uneven).unwrap_err();
    assert!(err.contains("'b'") && err.contains("expected 2"), "{err}");
  }
}
