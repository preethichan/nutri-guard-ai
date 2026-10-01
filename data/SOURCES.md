# Knowledge Base Sources

All documents in `data/raw/` are derived from real, authoritative, publicly available sources — no mock/synthetic data. Each file includes YAML frontmatter with the source organization, title, URL, and retrieval date. This manifest summarizes them for quick reference and auditing.

| File | Organization | Title | URL |
|---|---|---|---|
| `01_triglyceride_basics_nhlbi.md` | NHLBI, NIH | High Blood Triglycerides | https://www.nhlbi.nih.gov/health/high-blood-triglycerides |
| `02_triglyceride_levels_medlineplus.md` | MedlinePlus, NIH/NLM | Triglycerides | https://medlineplus.gov/triglycerides.html |
| `03_added_sugars_who_usda.md` | WHO; USDA/HHS | Guideline: Sugars intake for adults and children (2015); Dietary Guidelines for Americans 2020-2025 | https://www.who.int/publications/i/item/9789241549028 ; https://www.dietaryguidelines.gov |
| `04_saturated_trans_fat_who.md` | WHO | Saturated fatty acid and trans-fatty acid intake for adults and children: WHO guideline (2023) | https://www.who.int/publications/i/item/9789240073630 |
| `05_omega3_fatty_acids_nih_ods.md` | NIH Office of Dietary Supplements | Omega-3 Fatty Acids — Fact Sheet for Consumers | https://ods.od.nih.gov/factsheets/Omega3FattyAcids-Consumer/ |
| `06_alcohol_and_refined_carbs_aha.md` | American Heart Association | Triglycerides and Cardiovascular Disease (2011); AHA Dietary Guidelines; 2021 Dietary Guidance to Improve Cardiovascular Health | https://www.ahajournals.org/doi/10.1161/CIR.0b013e3182160726 |
| `07_fiber_and_dietary_pattern_limits_usda.md` | USDA/HHS | Dietary Guidelines for Americans, 2020-2025 (fiber, added sugar, saturated fat, sodium, alcohol limits) | https://www.dietaryguidelines.gov |
| `08_secondary_causes_and_escalation.md` | NHLBI, MedlinePlus, AHA (compiled) | Secondary/medical causes of high triglycerides and escalation guidance | (see individual sources above) |

## Notes on sourcing methodology
- All content was fetched live from the official organization websites (or their NCBI Bookshelf / government-hosted mirrors) and is quoted/paraphrased with attribution — not fabricated.
- Clinical reference ranges (e.g., triglyceride mg/dL categories) are quoted directly from NHLBI/MedlinePlus to avoid any drift/hallucination risk when the RAG system answers questions involving numbers.
- Where two sources cover similar ground (e.g., WHO's "free sugars" vs. USDA's "added sugars", or AHA's older 2-drink/1-drink guidance vs. its 2021 "if you don't drink, don't start" statement), **both are preserved and attributed separately** rather than merged — this is intentional. It gives the retrieval layer legitimate, sourced material to ground nuanced or evolving guidance, and gives the groundedness/faithfulness guardrail real examples of "answer must cite which source/version it's using" scenarios.
- Each document ends with a **"Scope note"** section marking where dietary guidance gives way to medical/clinical judgment. These are the seeds for the unintended-use / scope-boundary guardrail built later in this project.

## Corpus size
8 documents, ~800–900 lines total of real, sourced reference content — intentionally kept small (per project scope) but legitimate, to make retrieval, chunking, and guardrail behavior easy to inspect and debug.
