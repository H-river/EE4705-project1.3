| set | trial | claimed | actual | outcome | error_code | replans | llm_calls | wall_s | attribution (rule) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| e2e_smoke | smoke_1_standard | False | False | ERROR |  | 1 | 20 | 70.3 | A | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].color: 'empty' not in enum ['gray', 'dark_red', 'blue', 'green', 'red',… |
| e2e_smoke | smoke_2_scene_variation | False | True | REFUSED | UNREACHABLE | 3 | 11 | 98.4 | A | task achieved but not claimed (verification/perception) |
| e2e_smoke | smoke_3_instruction_variation | False | False | ERROR |  | 1 | 3 | 44.1 | A | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].bbox: violates minItems=4 |
| e2e_smoke | smoke_4_search | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 44 | 115.3 | C | failed codes TIMEOUT,SEARCH_NOT_FOUND,SEARCH_NOT_FOUND +contact |
| e2e_smoke | smoke_5_clarification | False | False | LIMIT_EXCEEDED |  | 9 | 15 | 243.6 | unknown | last events: [{"type": "search_found", "sim_time": 3.689999999999815, "target": "stone"}, {"type": "limit", "sim_time": 3.689999999999815, "which": "max_total_p… |
| e2e_smoke | smoke_1_standard | False | False | ERROR |  | 1 | 1 | 5.7 | A | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].bbox: violates minItems=4 |
| e2e_smoke | smoke_3_instruction_variation | False | False | ERROR |  | 1 | 1 | 5.2 | A | error: core.llm_client.SchemaError: VLM output invalid after one repair: $.detections[2].bbox: violates minItems=4 |
| e2e_smoke | smoke_1_standard | False | False | ERROR |  | 5 | 41 | 154.9 | A | error: core.llm_client.SchemaError: VLM output invalid after one repair: bbox must have positive width and height |
| e2e_smoke | smoke_3_instruction_variation | False | False | ERROR |  | 5 | 7 | 95.5 | A | error: core.llm_client.SchemaError: VLM output invalid after one repair: bbox must have positive width and height |
| e2e_v2 | c_2_01_stone | False | True | REFUSED | UNREACHABLE | 3 | 11 | 117.1 | A | task achieved but not claimed (verification/perception) |
| e2e_v2 | c_2_02_stone | False | True | REFUSED | UNREACHABLE | 3 | 13 | 121.4 | A | task achieved but not claimed (verification/perception) |
| e2e_v2 | c_2_03_stone | False | False | LIMIT_EXCEEDED | SEARCH_NOT_FOUND | 9 | 106 | 416.3 | unknown | last events: [{"type": "search_found", "sim_time": 41.59000000000486, "target": "stone"}, {"type": "limit", "sim_time": 41.59000000000486, "which": "max_total_p… |
| e2e_v2 | c_2_04_stone | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 3 | 39 | 180.1 | unknown | last events: [{"type": "action", "sim_time": 16.496000000001736, "skill": "SEARCH", "target": "stone", "success": false, "error": "SEARCH_NOT_FOUND", "attempt":… |
| e2e_v2 | c_2_05_stone | False | False | LIMIT_EXCEEDED | TIMEOUT | 9 | 39 | 339.5 | C | failed codes TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT +contact |
| e2e_v2 | c_2_06_cube | False | False | LIMIT_EXCEEDED |  | 9 | 17 | 170.0 | unknown | last events: [{"type": "search_found", "sim_time": 4.155999999999764, "target": "cube"}, {"type": "limit", "sim_time": 4.155999999999764, "which": "max_total_pl… |
| e2e_v2 | c_2_07_cube | False | False | LIMIT_EXCEEDED |  | 9 | 16 | 263.9 | unknown | last events: [{"type": "search_found", "sim_time": 3.689999999999815, "target": "cube"}, {"type": "limit", "sim_time": 3.689999999999815, "which": "max_total_pl… |
| e2e_v2 | c_2_08_cube | False | True | REFUSED | UNREACHABLE | 5 | 14 | 179.8 | A | task achieved but not claimed (verification/perception) |
| e2e_v2 | c_2_09_bottle | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 2 | 33 | 111.3 | unknown | last events: [{"type": "action", "sim_time": 16.026000000001996, "skill": "SEARCH", "target": "bottle", "success": false, "error": "SEARCH_NOT_FOUND", "attempt"… |
| e2e_v2 | c_2_10_bottle | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 30 | 100.4 | unknown | last events: [{"type": "action", "sim_time": 15.556000000001864, "skill": "SEARCH", "target": "bottle", "success": false, "error": "SEARCH_NOT_FOUND", "attempt"… |
| fix_r1 | c_2_01_stone | True | True | CLAIMED_SUCCESS |  | 0 | 6 | 25.3 | - |  |
| fix_r1 | c_2_02_stone | True | True | CLAIMED_SUCCESS |  | 1 | 6 | 20.3 | - |  |
| fix_r1 | c_2_08_cube | False | True | REFUSED | UNREACHABLE | 5 | 7 | 157.9 | A | task achieved but not claimed (verification/perception) |
| fix_r1 | smoke_2_scene_variation | True | True | CLAIMED_SUCCESS |  | 0 | 6 | 19.7 | - |  |
| r2_smoke | smoke_1_standard | True | True | CLAIMED_SUCCESS |  | 0 | 12 | 63.7 | - |  |
| r2_smoke | smoke_2_scene_variation | True | True | CLAIMED_SUCCESS |  | 0 | 4 | 17.7 | - |  |
| r2_smoke | smoke_3_instruction_variation | True | True | CLAIMED_SUCCESS |  | 0 | 1 | 19.5 | - |  |
| r2_smoke | smoke_4_search | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 2 | 22.5 | C | failed codes SEARCH_NOT_FOUND,SEARCH_NOT_FOUND +contact |
| r2_smoke | smoke_5_clarification | True | True | CLAIMED_SUCCESS |  | 1 | 11 | 85.7 | - |  |
| r2_v2 | c_2_01_stone | True | True | CLAIMED_SUCCESS |  | 0 | 4 | 20.5 | - |  |
| r2_v2 | c_2_02_stone | True | True | CLAIMED_SUCCESS |  | 0 | 12 | 68.0 | - |  |
| r2_v2 | c_2_03_stone | False | False | LIMIT_EXCEEDED |  | 9 | 31 | 373.9 | unknown | last events: [{"type": "search_found", "sim_time": 15.978000000002005, "target": "red_region"}, {"type": "limit", "sim_time": 15.978000000002005, "which": "max_… |
| r2_v2 | c_2_04_stone | False | True | LIMIT_EXCEEDED | PLACE_FAILED | 9 | 20 | 316.7 | A | task achieved but not claimed (verification/perception) |
| r2_v2 | c_2_05_stone | False | False | LIMIT_EXCEEDED | TIMEOUT | 9 | 27 | 263.7 | C | failed codes TARGET_LOST,TIMEOUT +contact |
| r2_v2 | c_2_06_cube | False | True | ERROR | INTERNAL_ERROR | 0 | 6 | 70.9 | A | error: fatal executor error: INTERNAL_ERROR |
| r2_v2 | c_2_07_cube | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 26 | 77.5 | unknown | last events: [{"type": "action", "sim_time": 15.556000000001864, "skill": "SEARCH", "target": "cube", "success": false, "error": "SEARCH_NOT_FOUND", "attempt": … |
| r2_v2 | c_2_08_cube | True | True | CLAIMED_SUCCESS |  | 1 | 6 | 58.2 | - |  |
| r2_v2 | c_2_09_bottle | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 4 | 56 | 186.0 | C | failed codes TARGET_LOST,TIMEOUT,SEARCH_NOT_FOUND,SEARCH_NOT_FOUND +contact |
| r2_v2 | c_2_10_bottle | False | False | SEARCH_EXHAUSTED | SEARCH_NOT_FOUND | 1 | 4 | 37.5 | unknown | last events: [{"type": "action", "sim_time": 15.556000000001864, "skill": "SEARCH", "target": "bottle", "success": false, "error": "SEARCH_NOT_FOUND", "attempt"… |
| r2_fixA | c_2_03_stone | False | False | LIMIT_EXCEEDED |  | 9 | 0 | 4.1 | unknown | last events: [{"type": "search_found", "sim_time": 15.978000000002005, "target": "red_region"}, {"type": "limit", "sim_time": 15.978000000002005, "which": "max_… |
| r2_fixA | c_2_05_stone | False | False | LIMIT_EXCEEDED | TIMEOUT | 9 | 1 | 6.4 | C | failed codes TARGET_LOST,TIMEOUT +contact |
| r2_fixA | smoke_4_search | False | False | ERROR | SEARCH_FATAL | 1 | 5 | 30.8 | C | error: fatal executor error: SEARCH_FATAL |
| r2_fixB | c_2_03_stone | False | False | REFUSED | TARGET_LOST | 9 | 81 | 377.4 | C | failed codes SEARCH_NOT_FOUND,TARGET_LOST +contact |
| r2_fixB | c_2_05_stone | False | False | LIMIT_EXCEEDED | TIMEOUT | 9 | 32 | 328.2 | C | failed codes TARGET_LOST,TIMEOUT +contact |
| r2_fixB | smoke_4_search | False | False | ERROR | SEARCH_FATAL | 1 | 14 | 64.2 | C | error: fatal executor error: SEARCH_FATAL |
