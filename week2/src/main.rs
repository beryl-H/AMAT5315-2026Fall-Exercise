use std::process::ExitCode;

use md::cli::{self, Command};
use md::ops;

fn usage() -> ! {
    eprintln!("usage: md run [--temp T] [--dt DT] [--steps N] [--equil E] [--out FILE] [--seed S]");
    eprintln!("       md check FILE [--temp-tol T] [--drift-tol D] [--ks-tol K]");
    eprintln!("       md video FILE [--out OUT.mp4] [--fps F]");
    std::process::exit(2);
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = match cli::parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            usage();
        }
    };
    match command {
        Command::Run(cfg) => match ops::run_sim(&cfg) {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        },
        Command::Check(cfg) => match ops::check(&cfg) {
            Ok(report) => {
                print!("{}", ops::format_report(&report, &cfg));
                if report.pass {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        },
        Command::Video(cfg) => match ops::make_video(&cfg) {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        },
    }
}
