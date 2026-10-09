//! V4.1 wire and bind gates, model-free.
//!
//! The metadata block is synthetic but carries the artifact's own arrays,
//! read from `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf` over the Spark
//! mount on 2026-10-08:
//!
//! - compress_ratios: 43 entries, first 40 are 0,0 then 2 through L19 and 1
//!   from L20 (the trailing three are ignored, `core_validate.c:14-46`)
//! - kv_source_layers [2, 8, 14, 20], index_source_layers
//!   [2, 8, 14, 20, 24, 28, 32, 36], engram.layer_ids [1, 14]
//! - mtp: 3 towers, 128 experts
//!
//! The tensor list is the artifact's published 1000 names
//! (`fixtures/v41/artifact-names.txt`), so the catalog gate is exact set
//! equality in both directions, not a spot check.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use ds4_core::{
    bind_names_v41, shape_for_variant, BindNeed, BindPlan, GgufFile, Shape, TensorInfo,
    TensorInventory, V41Wire, V41WireError, Variant,
};

fn shape() -> Shape {
    shape_for_variant(Variant::DeepSeek41Flash)
}

// The artifact's metadata arrays (see the module comment).
const COMPRESS_RATIOS: &[i32] = &[
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
];
const KV_SOURCES: &[i32] = &[2, 8, 14, 20];
const INDEX_SOURCES: &[i32] = &[2, 8, 14, 20, 24, 28, 32, 36];
const ENGRAM_LAYERS: &[i32] = &[1, 14];

enum Val<'a> {
    U32(u32),
    ArrayI32(&'a [i32]),
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn write_gguf(path: &Path, kvs: &[(&str, Val<'_>)]) {
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, 0);
    put_u64(&mut buf, kvs.len() as u64);
    for (key, val) in kvs {
        put_str(&mut buf, key);
        match val {
            Val::U32(v) => {
                put_u32(&mut buf, 4);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::ArrayI32(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 5);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    while buf.len() < 32 {
        buf.push(0);
    }
    fs::write(path, buf).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-v41");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// The artifact's metadata, with the arrays overridable for the refusal cases.
fn artifact_gguf(name: &str, kv: &[i32], idx: &[i32], ratios: &[i32]) -> GgufFile {
    let path = tmp(name);
    write_gguf(
        &path,
        &[
            ("deepseek4.attention.compress_ratios", Val::ArrayI32(ratios)),
            ("deepseek4.attention.kv_source_layers", Val::ArrayI32(kv)),
            ("deepseek4.attention.index_source_layers", Val::ArrayI32(idx)),
            ("deepseek4.engram.layer_ids", Val::ArrayI32(ENGRAM_LAYERS)),
            ("deepseek4.mtp.tower_count", Val::U32(3)),
            ("deepseek4.mtp.expert_count", Val::U32(128)),
        ],
    );
    GgufFile::open(&path).unwrap()
}

fn artifact_wire() -> V41Wire {
    V41Wire::load(
        &artifact_gguf(
            "artifact.gguf",
            KV_SOURCES,
            INDEX_SOURCES,
            COMPRESS_RATIOS,
        ),
        &shape(),
    )
    .unwrap()
}

/// A synthetic inventory holding exactly `names`, one tensor each. Only the
/// name matters to the bind plan; offsets and dims are placeholders.
fn inventory_from(names: &[String]) -> TensorInventory {
    let tensors = names
        .iter()
        .enumerate()
        .map(|(i, name)| TensorInfo {
            name: name.clone(),
            ndim: 1,
            dim: [1, 0, 0, 0, 0, 0, 0, 0],
            typ: 0,
            rel_offset: i as u64 * 4,
            abs_offset: i as u64 * 4,
            elements: 1,
            bytes: 4,
            shard: 0,
        })
        .collect();
    TensorInventory {
        shards: Vec::new(),
        tensors,
        data_pos: 0,
        alignment: 32,
        page: 4096,
    }
}

fn artifact_names() -> Vec<String> {
    let raw = include_str!("fixtures/v41/artifact-names.txt");
    raw.lines().map(|s| s.to_string()).collect()
}

#[test]
fn wire_matches_the_artifacts_wiring() {
    let w = artifact_wire();
    assert_eq!(w.mtp_towers, 3);
    assert_eq!(w.mtp_experts, 128);
    assert_eq!(w.engram_layers, vec![1, 14]);
    // Sources: the nearest earlier kv/index source per compressing layer.
    assert_eq!(w.kv_source_of[2], 2);
    assert_eq!(w.kv_source_of[7], 2);
    assert_eq!(w.kv_source_of[19], 14);
    assert_eq!(w.kv_source_of[24], 20);
    assert_eq!(w.kv_source_of[39], 20);
    assert_eq!(w.index_source_of[24], 24);
    assert_eq!(w.index_source_of[27], 24);
    assert_eq!(w.index_source_of[39], 36);
    // Ratio-0 layers compress nothing and read nothing.
    assert_eq!(w.kv_source_of[0], -1);
    assert_eq!(w.kv_source_of[1], -1);
    assert_eq!(w.index_source_of[1], -1);
    assert_eq!(w.engram_index_of[1], 0);
    assert_eq!(w.engram_index_of[14], 1);
    assert_eq!(w.engram_index_of[13], -1);
    // The tower experts are the blob form in this artifact.
    let inv = inventory_from(&artifact_names());
    assert!(w.tower_uses_blob(&inv, 0));
    assert!(w.tower_uses_blob(&inv, 2));
}

#[test]
fn compressing_layer_without_a_source_is_refused() {
    let g = artifact_gguf("no-source.gguf", &[], &[], COMPRESS_RATIOS);
    let err = V41Wire::load(&g, &shape()).expect_err("ratio 2 with no kv source");
    assert_eq!(err, V41WireError::NoSourceBefore(2));
}

#[test]
fn kv_source_must_also_be_an_index_source() {
    // L4 is a kv source that never appears in index_source_layers.
    let mut ratios = vec![0i32; 40];
    ratios[4] = 1;
    let g = artifact_gguf("kv-not-index.gguf", &[2, 4], &[2], &ratios);
    let err = V41Wire::load(&g, &shape()).expect_err("kv source without index source");
    assert_eq!(err, V41WireError::KvSourceNotIndexSource(4));
}

#[test]
fn a_layer_id_outside_the_shape_is_refused() {
    let g = artifact_gguf("layer-range.gguf", &[2, 40], INDEX_SOURCES, COMPRESS_RATIOS);
    let err = V41Wire::load(&g, &shape()).expect_err("layer 40 in a 40-layer model");
    assert_eq!(err, V41WireError::LayerRange("deepseek4.attention.kv_source_layers", 40));
}

#[test]
fn a_non_v41_shape_is_refused() {
    let g = artifact_gguf("not-v41.gguf", KV_SOURCES, INDEX_SOURCES, COMPRESS_RATIOS);
    let err = V41Wire::load(&g, &shape_for_variant(Variant::Flash)).expect_err("not v4.1");
    assert_eq!(err, V41WireError::NotV41);
}

#[test]
fn bind_catalog_equals_the_artifact_tensor_set() {
    let wire = artifact_wire();
    let names = artifact_names();
    let inv = inventory_from(&names);
    let catalog = bind_names_v41(&shape(), &wire, &inv);

    let catalog_set: HashSet<&str> = catalog.iter().map(|n| n.name.as_str()).collect();
    let artifact_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
    let missing: Vec<_> = artifact_set.difference(&catalog_set).collect();
    let extra: Vec<_> = catalog_set.difference(&artifact_set).collect();
    assert!(missing.is_empty(), "artifact tensors not in catalog: {missing:?}");
    assert!(extra.is_empty(), "catalog names not in artifact: {extra:?}");
    assert_eq!(catalog.len(), 1000);
    assert!(catalog.iter().all(|n| n.need == BindNeed::Required));

    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    assert_eq!(plan.missing_required(), Vec::<&str>::new());
}

#[test]
fn per_expert_towers_swap_the_blob_for_every_expert() {
    let wire = artifact_wire();
    let mut names = artifact_names();
    names.retain(|n| !n.ends_with("ffn_exps_vq.blob") || !n.starts_with("mtp."));
    for t in 0..3 {
        for e in 0..128 {
            for suffix in ["gate.weight", "up.weight", "down.weight"] {
                names.push(format!("mtp.{t}.ffn_exp.{e}.{suffix}"));
            }
        }
    }
    let inv = inventory_from(&names);
    let catalog = bind_names_v41(&shape(), &wire, &inv);
    assert!(!catalog
        .iter()
        .any(|n| n.name == "mtp.0.ffn_exps_vq.blob"));
    assert!(catalog
        .iter()
        .any(|n| n.name == "mtp.2.ffn_exp.127.down.weight"));
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    assert_eq!(plan.missing_required(), Vec::<&str>::new());
}

#[test]
fn a_tower_with_neither_expert_form_is_refused() {
    let wire = artifact_wire();
    let mut names = artifact_names();
    names.retain(|n| !n.starts_with("mtp."));
    let inv = inventory_from(&names);
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    let missing = plan.missing_required();
    assert!(missing.contains(&"mtp.0.ffn_exp.0.gate.weight"));
    assert!(missing.contains(&"mtp.main_proj.weight"));
}
