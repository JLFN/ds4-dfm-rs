//! The product route's job list: one render request per line.
//!
//! The route is a draft, an upscale and a decode; running a list of them in one
//! process is what makes the second clip cost only its own work. This module
//! owns the file and nothing else: the engine owns residency, the CLI owns the
//! route and the device.
//!
//! Seven TAB-separated fields, in order:
//!
//! ```text
//! prompt <TAB> seed <TAB> steps <TAB> width <TAB> height <TAB> frames <TAB> output
//! ```
//!
//! A tab because a prompt carries commas, quotes and spaces, and because a
//! list has to be writable from a shell without an escape language. Blank
//! lines and lines whose first non-space byte is `#` are skipped, so a list
//! can carry its own command.

use std::fmt;
use std::path::PathBuf;

/// One request's own fields. The route, the arms and the checkpoint are the
/// caller's, stated once for the whole list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub prompt: String,
    pub seed: u64,
    pub steps: usize,
    pub width: i32,
    pub height: i32,
    pub frames: i32,
    pub output: PathBuf,
}

/// The field count the format has, in the order the doc comment names them.
pub const FIELDS: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobError(String);

impl JobError {
    fn new(detail: impl Into<String>) -> Self {
        Self(detail.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for JobError {}

/// Parse a whole list. The `source` is the file's own name, so a bad line
/// names itself: without it a refusal cannot be acted on.
pub fn parse(text: &str, source: &str) -> Result<Vec<Job>, JobError> {
    let mut jobs = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != FIELDS {
            return Err(JobError::new(format!(
                "{source}:{}: {FIELDS} TAB-separated fields expected (prompt, seed, \
                 steps, width, height, frames, output), found {}",
                index + 1,
                fields.len()
            )));
        }

        let prompt = fields[0].to_string();
        if prompt.trim().is_empty() {
            return Err(JobError::new(format!(
                "{source}:{}: the prompt is empty",
                index + 1
            )));
        }

        let output = fields[6].trim();
        if output.is_empty() {
            return Err(JobError::new(format!(
                "{source}:{}: the output path is empty",
                index + 1
            )));
        }

        jobs.push(Job {
            prompt,
            seed: number(source, index + 1, "seed", fields[1])?,
            steps: number(source, index + 1, "steps", fields[2])?,
            width: number(source, index + 1, "width", fields[3])?,
            height: number(source, index + 1, "height", fields[4])?,
            frames: number(source, index + 1, "frames", fields[5])?,
            output: PathBuf::from(output),
        });
    }

    if jobs.is_empty() {
        return Err(JobError::new(format!(
            "{source}: the list holds no request"
        )));
    }

    Ok(jobs)
}

fn number<T: std::str::FromStr>(
    source: &str,
    line: usize,
    what: &str,
    text: &str,
) -> Result<T, JobError> {
    text.trim().parse().map_err(|_| {
        JobError::new(format!(
            "{source}:{line}: {what} is not a number: {:?}",
            text.trim()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "jobs.txt";

    #[test]
    fn a_prompt_holds_spaces_and_quotes() {
        let text = "the \"fox\" trots, then stops\t42\t4\t672\t384\t124\tout.mp4\n";
        let jobs = parse(text, SOURCE).unwrap();
        assert_eq!(jobs[0].prompt, "the \"fox\" trots, then stops");
        assert_eq!(jobs[0].output, PathBuf::from("out.mp4"));
    }

    #[test]
    fn parses_a_full_line() {
        let text = "a red fox trots, through snow\t42\t4\t672\t384\t124\toutputs/a.mp4\n\
                    A second prompt.\t7\t2\t512\t512\t22\toutputs/b.mp4\n";
        let jobs = parse(text, SOURCE).unwrap();

        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].prompt, "a red fox trots, through snow");
        assert_eq!(jobs[0].seed, 42);
        assert_eq!(jobs[0].steps, 4);
        assert_eq!(jobs[0].width, 672);
        assert_eq!(jobs[0].height, 384);
        assert_eq!(jobs[0].frames, 124);
        assert_eq!(jobs[0].output, PathBuf::from("outputs/a.mp4"));
        assert_eq!(jobs[1].steps, 2);
    }

    #[test]
    fn skips_comments_blanks_and_crlf() {
        let text = "\r\n  # a note\r\n\t\n\
                    fox\t1\t2\t512\t512\t22\tout.mp4\r\n";
        let jobs = parse(text, SOURCE).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].seed, 1);
    }

    #[test]
    fn names_the_line_it_refuses() {
        let error = parse("fox\t1\t2\n", SOURCE).unwrap_err();
        let message = error.message();
        assert!(message.contains("jobs.txt:1"), "{message}");
        assert!(message.contains("found 3"), "{message}");

        let error = parse("fox\tx\t2\t512\t512\t22\tout.mp4\n", SOURCE).unwrap_err();
        let message = error.message();
        assert!(
            message.contains("jobs.txt:1: seed is not a number"),
            "{message}"
        );

        let error = parse("\t1\t2\t512\t512\t22\tout.mp4\n", SOURCE).unwrap_err();
        assert!(
            error.message().contains("prompt is empty"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn refuses_an_empty_list() {
        let error = parse("# nothing here\n\n", SOURCE).unwrap_err();
        assert!(
            error.message().contains("no request"),
            "{}",
            error.message()
        );
    }
}
