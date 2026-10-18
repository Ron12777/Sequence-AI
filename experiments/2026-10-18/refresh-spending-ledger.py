"""Read only Vast charges for the authorized overnight rental; suppress unrelated rows."""
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from decimal import Decimal
ID=56487252
CLI='/home/rohan-mistry/.local/bin/vastai'
def cli(args):
 p=subprocess.run([CLI,*args,'--raw'],text=True,capture_output=True,timeout=45)
 if p.returncode:raise RuntimeError('Vast read failed; raw response suppressed')
 return json.loads(p.stdout)
def extract(rows):
 for row in rows:
  if str(ID) in str(row.get('source','')):
   yield {k:row.get(k) for k in ('type','source','amount','start','end')} | {'items':[{k:x.get(k) for k in ('type','amount','start','end')} for x in row.get('items',[])]}
  else:yield from extract(row.get('items',[]))
def main():
 a=argparse.ArgumentParser();a.add_argument('--stop-utc');args=a.parse_args()
 out=Path('experiments/2026-10-18/spending-ledger.json');previous=json.loads(out.read_text()) if out.exists() else {}
 now=datetime.now(timezone.utc);start=datetime(2026,10,18,tzinfo=timezone.utc)
 rows=[];token=None
 for _ in range(10):
  query=['show','invoices-v1','--charges','--charge-type','instance','--start-date',str(int(start.timestamp())),'--end-date',str(int(now.timestamp())),'--limit','100','--format','tree']
  r=cli(query+(['--next-token',token] if token else []));assert r.get('success') is not False
  rows.extend(extract(r.get('results',[])));token=r.get('next_token')
  if not token:break
 if token:raise RuntimeError('Incomplete pagination')
 instances=cli(['show','instances']);facts=previous.get('rental_facts',{})
 for x in instances:
  if x.get('id')==ID:facts={k:x.get(k) for k in ['id','start_date','actual_status','dph_total','gpu_name','num_gpus']}
 result={'queried_utc':now.isoformat(),'instance_id':ID,'provider_reported_charges_usd':float(sum((Decimal(str(x['amount'])) for x in rows),Decimal(0))), 'provider_rows':rows,'rental_facts':facts,'verified_destruction_utc':args.stop_utc,'rental_present':any(x.get('id')==ID for x in instances),'final_invoice':False,'billing_limitation':'Provider rounded charges may lag settlement; charges only, not net account cash spend. Timestamp estimate excludes transfers.'}
 if facts.get('start_date'):
  stop=datetime.fromisoformat(args.stop_utc) if args.stop_utc else now
  hours=(stop.timestamp()-facts['start_date'])/3600
  result['estimated_gpu_plus_storage_usd']=hours*facts['dph_total'];result['rental_hours']=hours
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='provider_rows'}))
if __name__=='__main__':main()
