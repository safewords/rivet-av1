use std::io::Write;
fn main() {
    let tree = std::env::var("TREE").expect("TREE=path to a rivet-av1 tree");
    println!("cargo:rerun-if-env-changed=TREE");
    println!("cargo:rustc-check-cfg=cfg(has_within)");
    let itx = std::fs::read_to_string(format!("{tree}/src/dsp/itx.rs")).unwrap();
    println!("cargo:rerun-if-changed={tree}/src/dsp/itx.rs");
    if itx.contains("fn inverse_transform_2d_within") {
        println!("cargo:rustc-cfg=has_within");
    }
    let out = std::env::var("OUT_DIR").unwrap();
    let mut f = std::fs::File::create(format!("{out}/mods.rs")).unwrap();
    writeln!(f, "#[path = \"{tree}/src/consts.rs\"] mod consts;\n#[path = \"{tree}/src/tables.rs\"] mod tables;\nmod dsp {{ #[path = \"{tree}/src/dsp/itx.rs\"] pub mod itx; #[cfg(target_arch = \"x86_64\")] pub fn avx2() -> bool {{ false }} }}\nuse dsp::itx;").unwrap();
}
