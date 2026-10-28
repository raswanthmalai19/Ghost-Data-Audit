This is a screenshot of an internal tool used by an online shop's operations staff (an admin console, search console, analytics view, support inbox, backup manager, connector console, object-storage browser, export centre, marketing audience view or similar).

Transcribe the records it shows. For every row or record that carries a resource identifier of the form `prefix_number` (for example `account_5762`, `search_428`, `support_1034`, `data_flow_1121`, `backup_9287`, `export_1260`, `analytics_7438`, `audit_record_003`, `marketing_001`), return one entry with:

- `resource_id`: the identifier exactly as printed;
- `emails`: every email address printed in the same row or in the detail panel that belongs to that record;
- `user_ids`: any internal customer references of the form `user_<number>` printed for that record;
- `status_text`: the status, mode, tier or state words shown for that record (for example `Ready`, `Continuous`, `Healthy`, `Archive`, `Live`, `Active`, `Retained reference`, `Valid`);
- `is_selected`: true when the record is the one highlighted or opened in the detail panel.

Set `screen_type` to a short description of the tool. Copy identifiers and email addresses character for character; do not guess characters you cannot read, and do not invent records that are not visible. Report only what is printed on the screen: do not judge whether any data should be deleted.
