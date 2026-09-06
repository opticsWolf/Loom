//! Material recipes: `MaterialSpec` + `evaluate` over the native kernels.
//!
//! Rust owns the recipes end-to-end (§9.3B): the same
//! `navette-materials` kernels Python calls, with the dispatch
//! transliterated arm-for-arm (defaults, aliases, nesting, error texts).
//! Params stay a raw value map (`BTreeMap<String, Value>`) — plain data
//! like Python's dicts, validated per model at use. Deliberate gains:
//! - `table_nk`'s grid-size assert becomes a `Result` (no panic);
//! - partial states fail at load, not at draw time.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// All known models, in canonical order (mirrors Python `MODELS`).
pub const MODELS: [&str; 23] = [
  "Konstant",
  "Table",
  "Cauchy",
  "CauchyUrbach",
  "Sellmeier",
  "SellmeierUrbach",
  "Lorentz",
  "Drude",
  "DrudeLorentz",
  "CodyLorentz",
  "ForouhiBloomerSingle",
  "ForouhiBloomerMulti",
  "ForouhiBloomerMetal",
  "ForouhiBloomerMetal2021",
  "TaucLorentz",
  "UBF",
  "Bruggeman",
  "MaxwellGarnett",
  "Looyenga",
  "Lichtenecker",
  "MoriTanaka",
  "PowerLaw",
  "Roughness",
];

/// A material as data: model name plus plain params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialSpec {
  pub model: String,
  pub params: BTreeMap<String, Value>,
}

impl MaterialSpec {
  pub fn new(model: impl Into<String>, params: BTreeMap<String, Value>) -> Self {
    Self { model: model.into(), params }
  }

  /// Evaluate to complex `n + ik` on `wavelengths` (must be non-empty).
  pub fn evaluate(&self, wavelengths: &[f64]) -> Result<Vec<Complex64>, String> {
    if wavelengths.is_empty() {
      return Err("wavelength_nm must be a non-empty 1-D array".to_string());
    }
    let wl = Array1::from_vec(wavelengths.to_vec());
    let w = wl.view();
    let p = &self.params;
    let m = self.model.as_str();
    let out: Array1<Complex64> = match m {
      "Konstant" => {
        let n = req_one(p, m, "n")?;
        navette_materials::table::konstant_nk(w, n, opt_float(p, "k", 0.0))
      }
      "Table" => {
        let (gw, nv) = req_table(p, m, "n_data")?;
        let kv = match p.get("k_data") {
          None => None,
          Some(_) => Some(req_table(p, m, "k_data")?.1),
        };
        for key in ["interpolation_type_n", "interpolation_type_k"] {
          if let Some(Value::String(s)) = p.get(key)
            && s != "linear" {
              return Err(format!(
                "Table {key}={s:?} unsupported: native core is linear-only, resample the table offline"
              ));
            }
        }
        if gw.len() < 2 {
          return Err(format!("table_nk needs at least 2 grid points, got {}", gw.len()));
        }
        navette_materials::table::table_nk(
          w,
          Array1::from_vec(gw).view(),
          Array1::from_vec(nv).view(),
          kv.map(Array1::from_vec).as_ref().map(|a| a.view()),
          opt_float(p, "n_factor", 1.0),
          opt_float(p, "k_factor", 1.0),
        )
      }
      "Cauchy" => {
        let v = req_float(p, m, &["A", "B", "C"])?;
        navette_materials::cauchy::cauchy_nk(w, v[0], v[1], v[2])
      }
      "CauchyUrbach" => {
        let v = req_float(p, m, &["A", "B", "C", "alpha0", "Eu", "lambda_g"])?;
        navette_materials::cauchy::cauchy_urbach_nk(w, v[0], v[1], v[2], v[3], v[4], v[5])
      }
      "Sellmeier" => {
        let v = req_float(p, m, &["B1", "C1", "B2", "C2", "B3", "C3"])?;
        navette_materials::sellmeier::sellmeier_nk(w, v[0], v[1], v[2], v[3], v[4], v[5])
      }
      "SellmeierUrbach" => {
        let v = req_float(p, m, &["B1", "C1", "B2", "C2", "B3", "C3", "alpha0", "Eu", "lambda_g"])?;
        navette_materials::sellmeier::sellmeier_urbach_nk(
          w, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
        )
      }
      "Lorentz" => {
        let osc = req_osc(p, m, "osc", 3)?;
        navette_materials::lorentz::lorentz_nk(w, osc.view(), opt_float(p, "epsilon_inf", 1.0))
      }
      "Drude" => {
        let v = req_float(p, m, &["omega_p", "gamma", "epsilon_inf"])?;
        navette_materials::drude::drude_nk(w, v[0], v[1], v[2])
      }
      "DrudeLorentz" => {
        let wp = opt_alias(p, &["omega_p", "wp"], 0.0);
        let gamma = opt_alias(p, &["gamma_drude", "gamma"], 0.0);
        let osc = req_osc(p, m, "osc", 3)?;
        navette_materials::drude::drude_lorentz_nk(
          w,
          wp,
          gamma,
          opt_float(p, "epsilon_inf", 1.0),
          osc.view(),
        )
      }
      "CodyLorentz" => {
        let v = req_float(p, m, &["Eg", "Et", "Eu"])?;
        let osc = req_osc(p, m, "osc", 4)?;
        navette_materials::cody_lorentz::cody_lorentz_nk(
          w,
          v[0],
          v[1],
          v[2],
          osc.view(),
          opt_float(p, "epsilon_inf", 1.0),
        )
        .map_err(|e| format!("CodyLorentz: {e}"))?
      }
      "ForouhiBloomerSingle" | "ForouhiBloomerMulti" => {
        let ib = req_osc(p, m, "ib", 4)?;
        navette_materials::forouhi_bloomer::fb_interband_nk(
          w,
          opt_float(p, "n_inf", 1.0),
          ib.view(),
        )
      }
      "ForouhiBloomerMetal" | "ForouhiBloomerMetal2021" => {
        let ib = req_osc(p, m, "ib", 4)?;
        let fe = req_fe(p)?;
        navette_materials::forouhi_bloomer::fb_metal_nk(
          w,
          opt_float(p, "n_inf", 1.0),
          Array1::from_vec(fe).view(),
          ib.view(),
        )
      }
      "TaucLorentz" => {
        let eg = req_one(p, m, "Eg")?;
        let osc = req_osc(p, m, "osc", 3)?;
        navette_materials::tauc_lorentz::tauc_lorentz_nk(
          w,
          eg,
          osc.view(),
          opt_float(p, "epsilon_inf", 1.0),
        )
        .map_err(|e| format!("TaucLorentz: {e}"))?
      }
      "UBF" => {
        let osc = req_ubf(p)?;
        navette_materials::ubf::ubf_nk(w, osc.view(), opt_float(p, "epsilon_inf", 1.0))
          .map_err(|e| format!("UBF: {e}"))?
      }
      "Bruggeman" | "MaxwellGarnett" | "Looyenga" | "Lichtenecker" | "MoriTanaka" | "PowerLaw" => {
        let host = req_spec(p, "host")?.evaluate(wavelengths)?;
        let incl = req_spec(p, "inclusion")?.evaluate(wavelengths)?;
        let f = req_one(p, m, "fraction")?;
        let (incl_a, host_a) = (Array1::from_vec(incl), Array1::from_vec(host));
        let (hi, hh) = (incl_a.view(), host_a.view());
        // NOTE: inclusion-first arg order mirrors Python exactly.
        let eps = match m {
          "Bruggeman" => navette_materials::ema::bruggeman(hi, hh, f, 100, 1e-9),
          "MaxwellGarnett" => navette_materials::ema::maxwell_garnett(hi, hh, f),
          "Looyenga" => navette_materials::ema::looyenga(hi, hh, f),
          "Lichtenecker" => navette_materials::ema::lichtenecker(hi, hh, f),
          "MoriTanaka" => {
            navette_materials::ema::mori_tanaka(hi, hh, f, opt_float(p, "L", 1.0 / 3.0))
          }
          _ => navette_materials::ema::general_power_law(hi, hh, f, opt_float(p, "alpha", 0.5)),
        };
        navette_materials::ema::eps_to_nk(eps.view())
      }
      "Roughness" => {
        let bottom = req_spec(p, "bottom")?.evaluate(wavelengths)?;
        let top = req_spec(p, "top")?.evaluate(wavelengths)?;
        let (bottom_a, top_a) = (Array1::from_vec(bottom), Array1::from_vec(top));
        let eps = navette_materials::ema::roughness_interface(bottom_a.view(), top_a.view());
        navette_materials::ema::eps_to_nk(eps.view())
      }
      other => {
        return Err(format!("Unknown material model {other:?}. Available: {MODELS:?}"));
      }
    };
    Ok(out.to_vec())
  }
}

fn missing_err(model: &str, names: &[&str]) -> String {
  format!("{model} missing params: [{}]", names.join(", "))
}

fn req_float(p: &BTreeMap<String, Value>, model: &str, names: &[&str]) -> Result<Vec<f64>, String> {
  let mut missing = Vec::new();
  let mut out = Vec::with_capacity(names.len());
  for n in names {
    match p.get(*n).and_then(|v| v.as_f64()) {
      Some(v) => out.push(v),
      None if !p.contains_key(*n) => missing.push(*n),
      None => return Err(format!("{model} param '{n}' must be numeric")),
    }
  }
  if missing.is_empty() {
    Ok(out)
  } else {
    Err(missing_err(model, &missing))
  }
}

fn req_one(p: &BTreeMap<String, Value>, model: &str, name: &str) -> Result<f64, String> {
  Ok(req_float(p, model, &[name])?[0])
}

fn opt_float(p: &BTreeMap<String, Value>, name: &str, default: f64) -> f64 {
  p.get(name).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// First present alias wins (mirrors `p.get("omega_p", p.get("wp", 0.0))`).
fn opt_alias(p: &BTreeMap<String, Value>, names: &[&str], default: f64) -> f64 {
  names.iter().find_map(|n| p.get(*n).and_then(|v| v.as_f64())).unwrap_or(default)
}

fn as_num_vec(v: &Value) -> Option<Vec<f64>> {
  v.as_array()?.iter().map(|x| x.as_f64()).collect()
}

/// `(wavelengths, values)` pair (mirrors the `(wavelengths, values)` tuples).
fn req_table(
  p: &BTreeMap<String, Value>,
  model: &str,
  name: &str,
) -> Result<(Vec<f64>, Vec<f64>), String> {
  let err = || format!("{model} {name} needs a (wavelengths, values) pair");
  let pair = p.get(name).and_then(|v| v.as_array()).ok_or_else(err)?;
  if pair.len() != 2 {
    return Err(err());
  }
  let x = as_num_vec(&pair[0]).ok_or_else(err)?;
  let y = as_num_vec(&pair[1]).ok_or_else(err)?;
  if x.len() != y.len() || x.is_empty() {
    return Err(err());
  }
  Ok((x, y))
}

/// Oscillator rows of fixed width (mirrors `_as_osc_array` incl. errors).
fn req_osc(
  p: &BTreeMap<String, Value>,
  model: &str,
  name: &str,
  width: usize,
) -> Result<Array2<f64>, String> {
  let what = format!("{model} {name}");
  let rows = p
    .get(name)
    .and_then(|v| v.as_array())
    .ok_or_else(|| format!("{what} needs an (N, {width}) array, got missing"))?;
  if rows.is_empty() {
    return Err(format!("{what} needs at least one oscillator"));
  }
  let mut flat = Vec::with_capacity(rows.len() * width);
  for r in rows {
    let row = as_num_vec(r).unwrap_or_default();
    if row.len() != width {
      return Err(format!("{what} needs an (N, {width}) array, got a row of width {}", row.len()));
    }
    flat.extend(row);
  }
  Array2::from_shape_vec((rows.len(), width), flat)
    .map_err(|e| format!("{what}: bad oscillator shape ({e})"))
}

/// `fe` triple: explicit `A_fe/B_fe/C_fe` or an `fe` array (mirrors Python).
fn req_fe(p: &BTreeMap<String, Value>) -> Result<Vec<f64>, String> {
  const MSG: &str = "ForouhiBloomerMetal needs fe (A_fe, B_fe, C_fe)";
  if ["A_fe", "B_fe", "C_fe"].iter().all(|k| p.contains_key(*k)) {
    let v = req_float(p, "ForouhiBloomerMetal", &["A_fe", "B_fe", "C_fe"])?;
    return Ok(v);
  }
  match p.get("fe").and_then(as_num_vec) {
    Some(v) if v.len() == 3 => Ok(v),
    _ => Err(MSG.to_string()),
  }
}

/// Nested spec (`host`/`inclusion`/`bottom`/`top`); missing → composite error.
fn req_spec(p: &BTreeMap<String, Value>, what: &str) -> Result<MaterialSpec, String> {
  let v = p.get(what).ok_or_else(|| format!("Composite material missing '{what}' spec"))?;
  if v.is_null() {
    return Err(format!("Composite material missing '{what}' spec"));
  }
  serde_json::from_value::<MaterialSpec>(v.clone())
    .map_err(|e| format!("Composite material '{what}' spec malformed ({e})"))
}

/// UBF oscillator dicts → kernel rows `[Eg, Ec, 1/Eu, A, Gamma, gamma]`.
fn req_ubf(p: &BTreeMap<String, Value>) -> Result<Array2<f64>, String> {
  let rows = p
    .get("osc")
    .and_then(|v| v.as_array())
    .ok_or_else(|| "UBF osc needs a list of oscillator dicts, got missing".to_string())?;
  if rows.is_empty() {
    return Err("UBF osc needs at least one oscillator".to_string());
  }
  let mut flat = Vec::with_capacity(rows.len() * 6);
  for (i, r) in rows.iter().enumerate() {
    let get = |k: &str| {
      r.get(k)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("UBF oscillator {i} missing key '{k}'"))
    };
    let (eg, ec, eu, a, gamma, gm) =
      (get("Eg")?, get("Ec")?, get("Eu")?, get("A")?, get("Gamma")?, get("gamma")?);
    if eu <= 0.0 {
      return Err(format!("UBF oscillator {i}: Eu must be > 0"));
    }
    flat.extend([eg, ec, 1.0 / eu, a, gamma, gm]);
  }
  Array2::from_shape_vec((rows.len(), 6), flat)
    .map_err(|e| format!("UBF: bad oscillator shape ({e})"))
}

#[cfg(test)]
mod tests {
  use super::*;

  const WL: [f64; 3] = [400.0, 800.0, 1600.0];

  fn spec(model: &str, params: Value) -> MaterialSpec {
    MaterialSpec {
      model: model.to_string(),
      params: serde_json::from_value(params).unwrap(),
    }
  }

  fn close(got: &[Complex64], want: &[[f64; 2]]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
      assert!((g.re - w[0]).abs() < 1e-12, "re {g} vs {w:?}");
      assert!((g.im - w[1]).abs() < 1e-12, "im {g} vs {w:?}");
    }
  }

  fn konst(n: f64, k: f64) -> Value {
    serde_json::json!({"model": "Konstant", "params": {"n": n, "k": k}})
  }

  /// Oracle twins: every model bit-tracks Python `evaluate` (values
  /// captured live from Python on WL; same kernels, same inputs).
  #[test]
  fn all_models_match_python() {
    assert_eq!(MODELS.len(), 23);
    let table = |w: Vec<f64>, v: Vec<f64>| serde_json::json!([w, v]);
    let cases: &[(&str, Value, [[f64; 2]; 3])] = &[
      ("Konstant", serde_json::json!({"n": 2.0, "k": 0.1}),
       [[2.0, 0.1], [2.0, 0.1], [2.0, 0.1]]),
      ("Table", serde_json::json!({"n_data": table(vec![300., 500., 800., 1200.], vec![2.6, 2.5, 2.4, 2.35]),
        "k_data": table(vec![300., 500., 800., 1200.], vec![0.02, 0.01, 0.005, 0.002])}),
       [[2.55, 0.015], [2.4, 0.005], [2.35, 0.002]]),
      ("Cauchy", serde_json::json!({"A": 1.5, "B": 0.01, "C": 0.0001}),
       [[1.56640625, 0.0], [1.515869140625, 0.0], [1.503921508789062, 0.0]]),
      ("CauchyUrbach", serde_json::json!({"A": 2.5, "B": 0.02, "C": 0.0005, "alpha0": 1e4, "Eu": 0.05, "lambda_g": 400.0}),
       [[2.64453125, 0.0], [2.532470703125, 2.200223592879712e-17], [2.507888793945312, 8.180694225817397e-24]]),
      ("Sellmeier", serde_json::json!({"B1": 1.0396, "C1": 0.0060, "B2": 0.2318, "C2": 0.0200, "B3": 1.0105, "C3": 103.56}),
       [[1.530834591147765, 0.0], [1.510772050857463, 0.0], [1.500018269952839, 0.0]]),
      ("SellmeierUrbach", serde_json::json!({"B1": 1.0396, "C1": 0.0060, "B2": 0.2318, "C2": 0.0200, "B3": 1.0105, "C3": 103.56, "alpha0": 1e4, "Eu": 0.05, "lambda_g": 400.0}),
       [[1.530834591147765, 0.0], [1.510772050857463, 2.200223592879712e-17], [1.500018269952839, 8.180694225817397e-24]]),
      ("Lorentz", serde_json::json!({"osc": [[3.0, 0.2, 0.5], [4.5, 0.1, 0.7]], "epsilon_inf": 1.0}),
       [[1.153731208791381, 1.621435145900531], [1.573151551249795, 1.2352353127448e-02], [1.502318258686829, 4.235632045690505e-03]]),
      ("Drude", serde_json::json!({"omega_p": 2.5, "gamma": 0.3, "epsilon_inf": 3.5}),
       [[1.68992544212249, 1.845590373858986e-02], [1.023761253547352, 2.371197698163733e-01], [7.118613045903379e-01, 2.461407418588736]]),
      ("DrudeLorentz", serde_json::json!({"omega_p": 2.5, "gamma": 0.3, "epsilon_inf": 3.5, "osc": [[3.0, 0.2, 0.5]]}),
       [[1.239022421780916, 1.519350784807504], [1.308238569034786, 1.977760652565055e-01], [7.444845390678393e-01, 2.360185402641237]]),
      ("CodyLorentz", serde_json::json!({"Eg": 1.5, "Et": 0.5, "Eu": 0.2, "osc": [[2.0, 0.5, 0.3, 25.0]], "epsilon_inf": 1.0}),
       [[9.999931254129154e-01, 2.933688920744446e-05], [1.00007490300154, 1.665040163139109e-07], [1.000046511480627, 0.0]]),
      ("ForouhiBloomerSingle", serde_json::json!({"ib": [[3.0, 0.1, 6.0, 12.0]], "n_inf": 1.5}),
       [[1.505731739947586, 3.296148880115328e-04], [1.450778367057924, 0.0], [1.451528702750885, 0.0]]),
      ("ForouhiBloomerMulti", serde_json::json!({"ib": [[3.0, 0.1, 6.0, 12.0], [4.0, 0.2, 5.0, 10.0]], "n_inf": 1.5}),
       [[1.245603303142273, 3.296148880115328e-04], [1.169425681065667, 0.0], [1.23904677648164, 0.0]]),
      ("ForouhiBloomerMetal", serde_json::json!({"ib": [[3.0, 0.1, 6.0, 12.0]], "fe": [8.0, 0.1, 0.3], "n_inf": 1.5}),
       [[2.926028938849746, 8.008632112202532], [4.170044483376079, 7.544483643532266], [5.792892498291914, 5.837037234100818]]),
      ("ForouhiBloomerMetal2021", serde_json::json!({"ib": [[3.0, 0.1, 6.0, 12.0]], "A_fe": 8.0, "B_fe": 0.1, "C_fe": 0.3, "n_inf": 1.5}),
       [[2.926028938849746, 8.008632112202532], [4.170044483376079, 7.544483643532266], [5.792892498291914, 5.837037234100818]]),
      ("TaucLorentz", serde_json::json!({"Eg": 2.0, "osc": [[30.0, 3.0, 1.0]], "epsilon_inf": 1.0}),
       [[1.780613483519783, 9.881532990951435e-01], [1.495882707729832, 0.0], [1.422104113544718, 0.0]]),
      ("UBF", serde_json::json!({"osc": [{"Eg": 2.0, "Ec": 3.0, "Eu": 0.5, "A": 20.0, "Gamma": 1.0, "gamma": 0.5}], "epsilon_inf": 1.0}),
       [[2.096898993413011, 2.176879958480976], [2.248023215134438, 1.696526148462792e-01], [2.054295667079059, 5.904283218991443e-02]]),
      ("Bruggeman", serde_json::json!({"host": konst(1.5, 0.0), "inclusion": konst(2.0, 0.1), "fraction": 0.3}),
       [[1.644192268867662, 2.757088420574801e-02]; 3]),
      ("MaxwellGarnett", serde_json::json!({"host": konst(1.5, 0.0), "inclusion": konst(2.0, 0.1), "fraction": 0.3}),
       [[1.641924124323112, 2.619214701179038e-02]; 3]),
      ("Looyenga", serde_json::json!({"host": konst(1.5, 0.0), "inclusion": konst(2.0, 0.1), "fraction": 0.3}),
       [[1.645097694510522, 2.810417585590106e-02]; 3]),
      ("Lichtenecker", serde_json::json!({"host": konst(1.5, 0.0), "inclusion": konst(2.0, 0.1), "fraction": 0.3}),
       [[1.635636368463494, 2.451596635220177e-02]; 3]),
      ("MoriTanaka", serde_json::json!({"host": konst(1.5, 0.0), "inclusion": konst(2.0, 0.1), "fraction": 0.3}),
       [[1.641924124323112, 2.619214701179038e-02]; 3]),
      ("PowerLaw", serde_json::json!({"host": konst(1.5, 0.0), "inclusion": konst(2.0, 0.1), "fraction": 0.3}),
       [[1.65, 0.03]; 3]),
      ("Roughness", serde_json::json!({"bottom": konst(1.5, 0.0), "top": konst(2.0, 0.1)}),
       [[1.744199090968062, 4.776419417604268e-02]; 3]),
    ];
    assert_eq!(cases.len(), 23);
    for (model, params, want) in cases {
      let s = spec(model, params.clone());
      let got = s.evaluate(&WL).unwrap_or_else(|e| panic!("{model}: {e}"));
      close(&got, want);
    }
  }

  #[test]
  fn dispatch_errors_match_python() {
    let wl = [1000.0];
    let bad = |m: &str, p: Value| spec(m, p).evaluate(&wl).unwrap_err();
    assert!(bad("Nope", serde_json::json!({})).contains("Unknown material model"));
    assert!(bad("Cauchy", serde_json::json!({"A": 1.5})).contains("missing params"));
    assert!(spec("Konstant", serde_json::json!({"n": 1.5})).evaluate(&[]).is_err());
    assert!(bad("Lorentz", serde_json::json!({})).contains("needs an (N, 3)"));
    assert!(bad("Lorentz", serde_json::json!({"osc": [[1.0, 2.0]]})).contains("width 2"));
    assert!(bad("Table", serde_json::json!({"n_data": [[400.0, 500.0], [2.0, 2.1]], "interpolation_type_n": "cubic"}))
      .contains("linear-only"));
    assert!(bad("Table", serde_json::json!({"n_data": [[[400.0]], [[2.0]]]}))
      .contains("needs a (wavelengths, values) pair"));
    assert!(bad("ForouhiBloomerMetal", serde_json::json!({"ib": [[1.0, 2.0, 3.0, 4.0]]}))
      .contains("needs fe"));
    assert!(bad("Bruggeman", serde_json::json!({"host": konst(1.5, 0.0), "fraction": 0.5}))
      .contains("missing 'inclusion'"));
    assert!(bad("UBF", serde_json::json!({"osc": [{"Eg": 2.0, "Ec": 3.0, "Eu": 0.0, "A": 1.0, "Gamma": 1.0, "gamma": 0.5}]})).contains("Eu must be > 0"));
  }
}
