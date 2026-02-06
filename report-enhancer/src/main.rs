//! Enhances Criterion.rs benchmark report:
//! 1. Adds mean time values to comparison tables
//! 2. Converts flat benchmark lists to comparison tables
//! 3. Colorizes violin plots with distinct colors
//!
//! Run after: cargo bench --bench comparison
//! Usage: cargo run -p report-enhancer

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

const VIOLIN_COLORS: &[&str] = &[
    "#1F78B4", "#FF7F00", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
];

fn format_time(ns: f64) -> String {
    if ns >= 1e9 {
        format!("{:.2} s", ns / 1e9)
    } else if ns >= 1e6 {
        format!("{:.2} ms", ns / 1e6)
    } else if ns >= 1e3 {
        format!("{:.2} µs", ns / 1e3)
    } else {
        format!("{:.1} ns", ns)
    }
}

fn collect_estimates(criterion_dir: &Path) -> HashMap<String, f64> {
    let mut estimates = HashMap::new();
    for entry in walkdir::WalkDir::new(criterion_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("estimates.json") {
            continue;
        }
        if path.components().any(|c| c.as_os_str() == "change") {
            continue;
        }
        if let Ok(content) = fs::read_to_string(path) {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(ns) = data
                    .get("mean")
                    .and_then(|m| m.get("point_estimate"))
                    .and_then(|v| v.as_f64())
                {
                    let parent = path.parent().unwrap_or(path.as_ref());
                    let bench_path = parent
                        .strip_prefix(criterion_dir)
                        .map(|p| p.to_string_lossy().replace('\\', "/"))
                        .unwrap_or_default();
                    let clean_key = bench_path
                        .strip_suffix("/new")
                        .or_else(|| bench_path.strip_suffix("/base"))
                        .unwrap_or(&bench_path)
                        .trim_end_matches('/')
                        .to_string();
                    if !clean_key.is_empty() {
                        estimates.insert(clean_key, ns);
                    }
                }
            }
        }
    }
    estimates
}

const VIOLIN_FONT_SCALE: f64 = 1.4;
const VIOLIN_FONT_SCALE_AXIS: f64 = 1.15; // smaller for left axis labels to avoid overlap with "Input"

fn colorize_violin(svg_path: &Path) {
    let Ok(content) = fs::read_to_string(svg_path) else { return };
    let mut result = content;
    let mut polygon_count = 0;
    while let Some(start) = result.find("fill=\"#1F78B4\"") {
        let color = VIOLIN_COLORS[(polygon_count / 2) % VIOLIN_COLORS.len()];
        result.replace_range(start..start + 14, &format!("fill=\"{color}\""));
        polygon_count += 1;
    }
    // Larger font in Violin Plot — skip left axis (rotate=Input, text-anchor=end=categories) to avoid overlap
    let re = regex::Regex::new(r#"font-size="([0-9.]+)""#).unwrap();
    let mut new_result = String::with_capacity(result.len());
    let mut last_end = 0;
    for cap in re.captures_iter(&result) {
        let full = cap.get(0).unwrap();
        let s = cap.get(1).unwrap().as_str();
        let pos = full.start();
        let ctx_start = pos.saturating_sub(80);
        let ctx_end = (full.end() + 80).min(result.len());
        let ctx = &result[ctx_start..ctx_end];
        let is_left_axis = ctx.contains("rotate(") || ctx.contains("text-anchor=\"end\"");
        new_result.push_str(&result[last_end..pos]);
        if let Ok(n) = s.parse::<f64>() {
            if is_left_axis {
                // Use smaller scale for left axis to avoid overlap; restore if already scaled
                let base = if n > 11.0 { n / VIOLIN_FONT_SCALE } else { n };
                let scaled = (base * VIOLIN_FONT_SCALE_AXIS * 10.).round() / 10.;
                new_result.push_str(&format!("font-size=\"{scaled}\""));
            } else if (7.0..=18.0).contains(&n) {
                let scaled = (n * VIOLIN_FONT_SCALE * 10.).round() / 10.;
                new_result.push_str(&format!("font-size=\"{scaled}\""));
            } else {
                new_result.push_str(full.as_str());
            }
        } else {
            new_result.push_str(full.as_str());
        }
        last_end = full.end();
    }
    new_result.push_str(&result[last_end..]);
    result = new_result;

    // Expand viewBox left so long labels (insert_clustered/BTreeSet) are not cut off
    let viewbox_re = regex::Regex::new(r#"viewBox="0 0 960 (\d+)"#).unwrap();
    if viewbox_re.is_match(&result) {
        let h = viewbox_re.captures(&result).and_then(|c| c.get(1)).map(|m| m.as_str()).unwrap_or("258");
        result = viewbox_re
            .replace(&result, format!(r#"viewBox="-120 0 1080 {}""#, h))
            .into_owned();
        result = result.replace("width=\"960\"", "width=\"1080\"");
    }
    // Fix malformed double-quote in viewBox (attributes construct error)
    let dq_re = regex::Regex::new(r#"viewBox="-120 0 1080 (\d+)"" "#).unwrap();
    result = dq_re.replace_all(&result, r#"viewBox="-120 0 1080 $1" "#).into_owned();

    let _ = fs::write(svg_path, result);
}

fn add_value_to_cell(cell: &str, estimates: &HashMap<String, f64>) -> String {
    let re = regex::Regex::new(r#"href="(\.\./[^"]+)"[^>]*>([^<]+)</a>"#).unwrap();
    if let Some(caps) = re.captures(cell) {
        let href = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let bench_id = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let path = href
            .trim_start_matches("../")
            .replace("\\report/index.html", "")
            .replace("/report/index.html", "")
            .replace('\\', "/");
        if let Some(&ns) = estimates.get(&path) {
            let time_str = format_time(ns);
            let short = bench_id.split('/').last().unwrap_or(bench_id);
            return format!(
                r#"<td><strong>{}</strong><br><a href="{}">{}</a></td>"#,
                time_str, href, short
            );
        }
    }
    cell.to_string()
}

fn enhance_index(index_path: &Path, estimates: &HashMap<String, f64>) {
    let Ok(mut content) = fs::read_to_string(index_path) else { return };

    // Add values to table cells matching: <td><a href="...">...</a></td>
    let re = regex::Regex::new(
        r#"<td><a href="(\.\./[^"]+)"[^>]*>([^<]+)</a></td>"#,
    )
    .unwrap();
    content = re
        .replace_all(&content, |caps: &regex::Captures| {
            let full = caps.get(0).map(|m| m.as_str()).unwrap_or("");
            add_value_to_cell(full, estimates)
        })
        .into_owned();

    // Convert flat lists to comparison tables
    let groups: &[(&str, &[(&str, &str)])] = &[
        (
            "contains",
            &[
                ("btreeset_existing", "existing"),
                ("btreeset_missing", "missing"),
                ("trie_existing", "existing"),
                ("trie_missing", "missing"),
            ],
        ),
        ("contains_clustered", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        ("insert_clustered", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        ("insert_random", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        ("iter_full", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        ("mixed_workload", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        ("predecessor", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        (
            "range_queries",
            &[
                ("trie_small_range", "small_range"),
                ("btreeset_small_range", "small_range"),
                ("trie_medium_range", "medium_range"),
                ("btreeset_medium_range", "medium_range"),
                ("trie_large_range", "large_range"),
                ("btreeset_large_range", "large_range"),
                ("trie_cross_cluster", "cross_cluster"),
                ("btreeset_cross_cluster", "cross_cluster"),
            ],
        ),
        ("range_sparse", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        (
            "remove",
            &[
                ("trie_sequential", "sequential"),
                ("btreeset_sequential", "sequential"),
                ("trie_sparse", "sparse"),
                ("btreeset_sparse", "sparse"),
            ],
        ),
        ("successor", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
        ("successor_sequential", &[("trie", "Trie"), ("btreeset", "BTreeSet")]),
    ];

    for (group_name, items) in groups {
        let block_start = content
            .find(&format!("<li><a href=\"../{group_name}"))
            .or_else(|| content.find(&format!("<li><a href=\"../{}", group_name.replace('_', ""))));
        let Some(block_start) = block_start else { continue };
        let ul_start = content[block_start..].find("<ul>");
        let Some(ul_start) = ul_start else { continue };
        let ul_start = block_start + ul_start;
        let ul_end = content[ul_start..].find("</ul>").map(|i| ul_start + i + 5).unwrap_or(ul_start);

        let table = build_comparison_table(group_name, items, estimates);
        let new_block = format!(
            r#"<li><a href="../{group_name}/report/index.html">{group_name}</a></li>
            <ul>
                <li>
                    {table}
                </li>
            </ul>"#
        );
        content.replace_range(block_start..ul_end, &new_block);
    }

    // Wider table columns, larger font
    if content.contains("border: 1px solid #888;") && !content.contains("min-width: 180px") {
        content = content.replace(
            "border: 1px solid #888;\n        }\n    </style>",
            "border: 1px solid #888;\n        }\n        th, td {\n            min-width: 180px;\n            padding: 8px 14px;\n            font-size: 16px;\n        }\n    </style>",
        );
    } else if content.contains("min-width: 180px") && !content.contains("font-size: 16px") {
        content = content.replace(
            "min-width: 180px;\n            padding: 8px 14px;\n        }",
            "min-width: 180px;\n            padding: 8px 14px;\n            font-size: 16px;\n        }",
        );
    }

    let _ = fs::write(index_path, content);
}

fn build_comparison_table(
    group: &str,
    items: &[(&str, &str)],
    estimates: &HashMap<String, f64>,
) -> String {
    let trie: Vec<_> = items.iter().filter(|(p, _)| p.contains("trie")).collect();
    let btree: Vec<_> = items.iter().filter(|(p, _)| p.contains("btreeset")).collect();

    if group == "range_queries" && trie.len() == 4 && btree.len() == 4 {
        let rows = ["small_range", "medium_range", "large_range", "cross_cluster"];
        let mut table = String::from("<table><tr><th></th><th>Trie</th><th>BTreeSet</th></tr>");
        for (i, row) in rows.iter().enumerate() {
            let t_key = format!("{}/{}", group, trie[i].0);
            let b_key = format!("{}/{}", group, btree[i].0);
            let t_val = estimates.get(&t_key).map(|&ns| format_time(ns)).unwrap_or_else(|| "—".into());
            let b_val = estimates.get(&b_key).map(|&ns| format_time(ns)).unwrap_or_else(|| "—".into());
            table.push_str(&format!(
                r#"<tr><th>{}</th><td><strong>{}</strong><br><a href="../{}/{}/report/index.html">{}</a></td><td><strong>{}</strong><br><a href="../{}/{}/report/index.html">{}</a></td></tr>"#,
                row,
                t_val,
                group,
                trie[i].0,
                trie[i].0,
                b_val,
                group,
                btree[i].0,
                btree[i].0
            ));
        }
        table.push_str("</table>");
        return table;
    }

    if trie.len() == btree.len() && trie.len() > 1 {
        let mut table = format!(
            r#"<table><tr><th></th><th><a href="../{}/report/index.html">Trie</a></th><th><a href="../{}/report/index.html">BTreeSet</a></th></tr>"#,
            group, group
        );
        for (t, b) in trie.iter().zip(btree.iter()) {
            let row = t.1;
            let t_key = format!("{}/{}", group, t.0);
            let b_key = format!("{}/{}", group, b.0);
            let t_val = estimates.get(&t_key).map(|&ns| format_time(ns)).unwrap_or_else(|| "—".into());
            let b_val = estimates.get(&b_key).map(|&ns| format_time(ns)).unwrap_or_else(|| "—".into());
            table.push_str(&format!(
                r#"<tr><th>{}</th><td><strong>{}</strong><br><a href="../{}/{}/report/index.html">{}</a></td><td><strong>{}</strong><br><a href="../{}/{}/report/index.html">{}</a></td></tr>"#,
                row, t_val, group, t.0, t.0, b_val, group, b.0, b.0
            ));
        }
        table.push_str("</table>");
        return table;
    }

    if items.len() == 2 {
        let t_key = format!("{}/{}", group, items[0].0);
        let b_key = format!("{}/{}", group, items[1].0);
        let t_val = estimates.get(&t_key).map(|&ns| format_time(ns)).unwrap_or_else(|| "—".into());
        let b_val = estimates.get(&b_key).map(|&ns| format_time(ns)).unwrap_or_else(|| "—".into());
        return format!(
            r#"<table><tr><th>Trie</th><th>BTreeSet</th></tr><tr><td><strong>{}</strong><br><a href="../{}/{}/report/index.html">{}</a></td><td><strong>{}</strong><br><a href="../{}/{}/report/index.html">{}</a></td></tr></table>"#,
            t_val, group, items[0].0, items[0].0,
            b_val, group, items[1].0, items[1].0
        );
    }

    String::new()
}

/// Add responsive CSS to all report index.html so charts (violin, etc.) fit the page.
fn make_reports_fit_width(criterion_dir: &Path) {
    let responsive_css = "\n        .body { max-width: 100%; box-sizing: border-box; }\n        img { max-width: 100%; height: auto; }\n    ";
    for entry in walkdir::WalkDir::new(criterion_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("index.html") {
            continue;
        }
        if !path.parent().is_some_and(|p| p.file_name().and_then(|n| n.to_str()) == Some("report")) {
            continue;
        }
        if let Ok(mut content) = fs::read_to_string(path) {
            if content.contains("max-width: 100%") && content.contains("box-sizing: border-box") {
                continue;
            }
            if content.contains("</style>") {
                content = content.replace("</style>", &format!("{responsive_css}</style>"));
                let _ = fs::write(path, content);
            }
        }
    }
}

fn main() {
    // report-enhancer lives in report-enhancer/, target is at parent/target
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let criterion_dir = Path::new(&manifest_dir)
        .parent()
        .unwrap_or(Path::new(&manifest_dir))
        .join("target")
        .join("criterion");

    if !criterion_dir.exists() {
        eprintln!("Error: Criterion directory not found: {}", criterion_dir.display());
        eprintln!("Run 'cargo bench -p clustered-fast-trie --bench comparison' first.");
        std::process::exit(1);
    }

    println!("Collecting benchmark estimates...");
    let estimates = collect_estimates(&criterion_dir);
    println!("Found {} benchmark results.", estimates.len());

    let index_path = criterion_dir.join("report").join("index.html");
    if index_path.exists() {
        println!("Enhancing index.html with values and comparison tables...");
        enhance_index(&index_path, &estimates);
    }

    println!("Colorizing violin plots...");
    for entry in walkdir::WalkDir::new(&criterion_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "violin.svg" {
            colorize_violin(entry.path());
        }
    }

    println!("Making report pages fit width...");
    make_reports_fit_width(&criterion_dir);

    println!("Done! Open target/criterion/report/index.html");
}
