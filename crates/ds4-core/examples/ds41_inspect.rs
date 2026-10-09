//! Metadata-only inventory of a DeepSeek V4.1 artifact; never loads weights.
//!
//!   cargo run -p ds4-core --example ds41_inspect -- <model.gguf> [--tensors]
//!
//! Prints the header's identifying keys, the per-type tensor census, and the
//! V4.1-specific keys the engine reads (core_validate_v41.c:46-154). This is
//! the P2/P3 check of docs/deepseek41-port-plan.md: the Rust host must see the
//! same artifact the C engine sees before any forward work starts.
use ds4_core::{tensor_type_name, GgufFile, TensorInventory};
use std::collections::BTreeMap;
use std::path::Path;

/// The keys the engine reads for V4.1 (core_validate_v41.c), plus the plain
/// identity keys and the dims `select_shape_from_metadata` matches on. A
/// missing one here is a loader gap, not a curiosity.
const V41_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    // dims: what ds4_select_shape_from_metadata compares (core_shape_select.c:166)
    "deepseek4.block_count",
    "deepseek4.embedding_length",
    "deepseek4.vocab_size",
    "deepseek4.attention.head_count",
    "deepseek4.attention.head_count_kv",
    "deepseek4.attention.key_length",
    "deepseek4.attention.value_length",
    "deepseek4.rope.dimension_count",
    "deepseek4.attention.q_lora_rank",
    "deepseek4.attention.output_lora_rank",
    "deepseek4.attention.output_group_count",
    "deepseek4.expert_count",
    "deepseek4.expert_used_count",
    "deepseek4.expert_feed_forward_length",
    "deepseek4.expert_shared_count",
    "deepseek4.hash_layer_count",
    "deepseek4.attention.sliding_window",
    "deepseek4.attention.indexer.head_count",
    "deepseek4.attention.indexer.key_length",
    "deepseek4.attention.indexer.top_k",
    "deepseek4.hyper_connection.count",
    "deepseek4.hyper_connection.sinkhorn_iterations",
    // V4.1 additions (core_validate_v41.c:46-154)
    "deepseek4.context_length",
    "deepseek4.attention.kv_source_layers",
    "deepseek4.attention.index_source_layers",
    "deepseek4.attention.candidate.source_layer",
    "deepseek4.attention.candidate.topk_blocks",
    "deepseek4.attention.candidate.block_size",
    "deepseek4.engram.layer_ids",
    "deepseek4.engram.num_embeddings",
    "deepseek4.engram.max_ngram_size",
    "deepseek4.mtp.tower_count",
    "deepseek4.mtp.expert_count",
    "deepseek4.mtp.block_size",
    "deepseek4.mtp.expert_used_count",
    "deepseek4.mtp.noise_token_id",
    "deepseek4.mtp.markov_rank",
    "deepseek4.mtp.target_layers",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: ds41_inspect <model.gguf> [--tensors]")?;
    let list_tensors = args.next().as_deref() == Some("--tensors");
    let p = Path::new(&path);

    let g = GgufFile::open(p)?;
    let inv = TensorInventory::open(p)?;

    println!(
        "file: {} kv={} tensors={} data_pos={} alignment={}",
        path,
        g.kv_entries().len(),
        inv.tensors.len(),
        inv.data_pos,
        inv.alignment
    );

    // Per-type census: the types above 30 are this family's, so a census
    // shows at a glance whether 40-44 are parsed.
    let mut census: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    for t in &inv.tensors {
        let e = census.entry(t.typ).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.bytes;
    }
    println!("types:");
    for (typ, (count, bytes)) in &census {
        println!(
            "  {typ:>3} {:<10} tensors={count} bytes={bytes}",
            tensor_type_name(*typ)
        );
    }

    println!("keys:");
    for key in V41_KEYS {
        let seen = match g.get_string(key) {
            Some(v) => format!("str {:?}", String::from_utf8_lossy(v)),
            None => match g.get_u32(key) {
                Some(v) => format!("u32 {v}"),
                None => match g.get_u64_compat(key) {
                    Some(v) => format!("u64 {v}"),
                    None => match g.get_array(key) {
                        Some(arr) => format!("array len={}", arr.len),
                        None => "ABSENT".to_string(),
                    },
                },
            },
        };
        println!("  {key} = {seen}");
    }

    if list_tensors {
        println!("tensors:");
        for t in &inv.tensors {
            println!("  {}", t.dump_line());
        }
    }

    // Identification and validation: the two steps that decide whether this
    // host can accept the artifact at all.
    match ds4_core::identify_file(&g) {
        Ok(id) => {
            println!("identify: {}", id.identify_line());
            match ds4_core::validate_file(&g, &id.shape) {
                Ok(()) => println!("validate: ok"),
                Err(e) => println!("validate: {}", e.token()),
            }
        }
        Err(e) => println!("identify failed: {e}"),
    }
    Ok(())
}
