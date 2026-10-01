//! VAE format tool for the image engine (P0).
//!
//!     qwen-image-vae-gguf <qwen_image_2.1_vae_bf16.safetensors> <out.gguf>
//!
//! Converts the exporter's safetensors VAE to the decode-only GGUF the tree
//! reads, with the layout pinned by `ds4_core::qwen_image::vae_decode_contract`.
//! The tool is host-only: it touches no device and no FFI symbol.

fn usage() -> ! {
    eprintln!("usage: qwen-image-vae-gguf <source.safetensors> <out.gguf>");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        usage();
    }
    let src = std::path::Path::new(&args[0]);
    let out = std::path::Path::new(&args[1]);
    match ds4_core::qwen_image::convert::convert_vae(src, out) {
        Ok(report) => println!("{}", report.line(out)),
        Err(e) => {
            eprintln!("qwen-image-vae-gguf: {e}");
            std::process::exit(1);
        }
    }
}
