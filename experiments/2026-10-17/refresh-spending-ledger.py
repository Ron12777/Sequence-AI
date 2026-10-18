"""Read-only Vast billing refresh, retaining only four campaign rental IDs.

CLI responses stay in memory. Account identity, balance, keys, unrelated charge
rows, labels, and arbitrary metadata are neither printed nor saved. No mutation
commands are supported. The API uses billing-day filters, not rental timestamps.
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDS = (56294531, 56404543, 56406932, 56415071)
START = datetime(2026, 10, 16, 20, 37, tzinfo=timezone.utc)


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def cli(arguments):
    result = subprocess.run(
        ["vastai", *arguments, "--raw"],
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )
    if result.returncode:
        raise RuntimeError(
            f"Read-only Vast request failed with exit code {result.returncode}; raw response suppressed"
        )
    return json.loads(result.stdout)


def instance_id(row):
    source = str(row.get("source", ""))
    for identifier in IDS:
        if re.search(r"(?<!\d)" + str(identifier) + r"(?!\d)", source):
            return identifier
    return None


def sanitized_charge(row):
    result = {
        key: row[key]
        for key in ("type", "source", "amount", "start", "end")
        if key in row
    }
    result["items"] = [sanitized_charge(child) for child in row.get("items", [])]
    return result


def extract(rows):
    for row in rows:
        identifier = instance_id(row)
        if identifier is not None:
            yield {"instance_id": identifier, **sanitized_charge(row)}
        else:
            yield from extract(row.get("items", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/2026-10-17/spending-ledger-provisional.json",
    )
    parser.add_argument(
        "--final-stop-utc",
        help="Known verified destruction/stop timestamp for the final rental; otherwise estimates extend through query time",
    )
    args = parser.parse_args()
    queried = datetime.now(timezone.utc)
    day_start = START.replace(hour=0, minute=0)
    query = [
        "show",
        "invoices-v1",
        "--charges",
        "--charge-type",
        "instance",
        "--start-date",
        str(int(day_start.timestamp())),
        "--end-date",
        str(int(queried.timestamp())),
        "--limit",
        "100",
        "--format",
        "tree",
    ]
    charges, pages, token = [], 0, None
    for _ in range(10):
        response = cli(query + (["--next-token", token] if token else []))
        if response.get("success") is False:
            raise RuntimeError("Provider rejected billing read")
        charges.extend(extract(response.get("results", [])))
        pages += 1
        token = response.get("next_token")
        if not token:
            break
    if token:
        raise RuntimeError(
            "Pagination exceeded bounded read; incomplete ledger not saved"
        )
    campaign_path = ROOT / "experiments/2026-10-17/campaign.json"
    campaign_bytes = campaign_path.read_bytes()
    campaign = json.loads(campaign_bytes)
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    facts = previous.get("instance_start_and_rate_facts", {})
    for identifier in IDS:
        try:
            instance = cli(["show", "instance", str(identifier)])
            if isinstance(instance, dict) and "instances" in instance:
                instance = instance["instances"]
            if isinstance(instance, list):
                instance = next(
                    (item for item in instance if item.get("id") == identifier), None
                )
            if isinstance(instance, dict) and instance.get("id") == identifier:
                facts[str(identifier)] = {
                    key: instance[key]
                    for key in (
                        "id",
                        "start_date",
                        "actual_status",
                        "dph_base",
                        "dph_total",
                        "storage_cost",
                        "disk_space",
                        "gpu_name",
                        "num_gpus",
                    )
                    if key in instance
                }
                facts[str(identifier)]["read_utc"] = queried.isoformat()
            elif str(identifier) not in facts:
                facts[str(identifier)] = {
                    "historical_instance_details_unavailable": True
                }
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            if str(identifier) not in facts:
                facts[str(identifier)] = {
                    "historical_instance_details_unavailable": True
                }
    categories = {}
    totals = {str(identifier): Decimal(0) for identifier in IDS}
    for row in charges:
        totals[str(row["instance_id"])] += Decimal(str(row["amount"]))
        for child in row["items"]:
            kind = child["type"]
            categories[kind] = categories.get(kind, Decimal(0)) + Decimal(
                str(child["amount"])
            )
    migration = campaign["migration"]
    efficient = campaign["efficiency_migration"]
    multi = campaign["multigpu_plan"]
    final_facts = facts.get("56415071", {})
    last_start = (
        iso(final_facts["start_date"])
        if "start_date" in final_facts
        else multi["started_utc"]
    )
    final_end = args.final_stop_utc or queried.isoformat()
    # These are estimates from recorded timestamps, never substituted for the
    # provider's charge ledger. Successor creation is only an upper-bound proxy
    # for destruction when an exact preceding destruction time was not saved.
    segments = [
        (
            56294531,
            campaign["started_at_utc"],
            migration["old_destroyed_at_utc"].split(" ")[0],
            0.4681481481481482,
            "Campaign work start, not verified rental creation; destruction timestamp explicitly approximate.",
        ),
        (
            56404543,
            migration["new_created_utc"],
            efficient["created_utc"],
            migration["quoted_hourly_usd"],
            "Recorded creation; successor creation used as upper-bound end proxy, not verified destruction time.",
        ),
        (
            56406932,
            efficient["created_utc"],
            last_start,
            efficient["hourly_usd"],
            "Recorded creation; successor provider start used as upper-bound end proxy, not verified destruction time.",
        ),
        (
            56415071,
            last_start,
            final_end,
            multi["actual_hourly_usd"],
            "Provider start timestamp when available; quoted full GPU plus storage hourly rate; excludes transfers/adjustments.",
        ),
    ]
    estimates = []
    for identifier, begin, end, rate, note in segments:
        hours = (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            - datetime.fromisoformat(begin.replace("Z", "+00:00"))
        ).total_seconds() / 3600
        estimates.append(
            {
                "instance_id": identifier,
                "start_utc": begin,
                "end_utc": end,
                "hours": hours,
                "gpu_plus_storage_hourly_usd": rate,
                "estimated_gpu_plus_storage_usd": hours * rate,
                "actual_spending": False,
                "timing_limitation": note,
            }
        )
    report = {
        "status": "provisional_provider_charge_ledger",
        "queried_utc": queried.isoformat(),
        "requested_campaign_start_utc": START.isoformat(),
        "campaign_deadline_utc": "2026-10-17T22:00:00Z",
        "task_instances": list(IDS),
        "read_only": True,
        "unrelated_records_retained": False,
        "provider_reported_task_charges_usd": float(sum(totals.values())),
        "per_instance_provider_charges_usd": {
            key: float(value) for key, value in totals.items()
        },
        "provider_charge_categories_usd": {
            key: float(value) for key, value in categories.items()
        },
        "category_meaning": {
            "gpu": "GPU/compute rental charge",
            "disk": "storage rental charge",
            "bwd": "download traffic charge",
            "bwu": "upload traffic charge",
        },
        "credit_adjustments_usd": None,
        "credit_adjustments_scope": "Charges-only endpoint. Purchases, credit balance, refunds, transfers, promotional credits, and taxes are not queried or netted. No net account cash-spend claim.",
        "final_invoice": False,
        "billing_limitations": [
            "Provider raw rows report amounts rounded to three decimals; totals can change or lag until settlement.",
            "The endpoint filters aggregated billing days. Query starts at October 16 00:00 UTC to include the campaign's partial first day, then retains only the four task rentals.",
            "Charge start/end fields identify billing days, not exact rental creation/destruction. Precisely excluding any pre-20:37 activity on the first task rental is not possible from this response.",
            "Provider-reported charges and timestamp-derived estimates differ; estimates do not replace the provider ledger or imply unreported fees are zero.",
        ],
        "provider_query": {
            "argv": ["vastai", *query, "--raw"],
            "pages": pages,
            "pagination_complete": True,
            "kept_instance_ids": list(IDS),
        },
        "provider_source_rows": charges,
        "instance_start_and_rate_facts": facts,
        "timestamp_estimates": estimates,
        "estimated_gpu_plus_storage_total_usd": sum(
            row["estimated_gpu_plus_storage_usd"] for row in estimates
        ),
        "estimate_source_campaign_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
        "refresh_command": "/tmp/sequence-ai-venv/bin/python experiments/2026-10-17/refresh-spending-ledger.py --output experiments/2026-10-17/spending-ledger-provisional.json",
        "final_refresh_note": "Run the refresh command after verified destruction. Optionally append --final-stop-utc with the verified final stop/destruction timestamp to close the estimate segment; provider charges may still settle later.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queried_utc": report["queried_utc"],
                "provider_reported_task_charges_usd": report[
                    "provider_reported_task_charges_usd"
                ],
                "per_instance": report["per_instance_provider_charges_usd"],
                "categories": report["provider_charge_categories_usd"],
                "final_invoice": False,
            }
        )
    )


if __name__ == "__main__":
    main()
