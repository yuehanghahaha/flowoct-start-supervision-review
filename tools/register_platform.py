"""Register a mapping AFTER checking your own host home device/inode.

Inside your allocated slot: register_platform.py --run RUN --host-device DEV --host-inode INODE
This performs no CUDA operation, GPU request, resource release, or environment change.
"""
from pathlib import Path
import argparse,getpass,socket,json,datetime


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--host-device',type=int,required=True);p.add_argument('--host-inode',type=int,required=True)
    args=p.parse_args();home=Path.home().resolve();user=getpass.getuser()
    if home!=Path('/workspace') or socket.gethostname() not in (f'hpc-{user}-slot1',f'hpc-{user}-slot2'):
        raise RuntimeError('Enter your own allocated platform container first.')
    run=args.run.resolve()
    if home not in run.parents:raise ValueError('Output leaves the persistent user directory.')
    observed=home.stat();expected=[args.host_device,args.host_inode]
    if expected!=[observed.st_dev,observed.st_ino]:raise RuntimeError('Host/container mappings differ.')
    run.mkdir(parents=True,exist_ok=True)
    receipt={'account':user,'persistent_home_inode':expected,'container':socket.gethostname(),
             'verified_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    target=run/'platform_mapping.json'
    if target.exists() and json.loads(target.read_text())['persistent_home_inode']!=expected:
        raise RuntimeError('Do not overwrite an incompatible persistent mapping.')
    target.write_text(json.dumps(receipt,indent=2)+'\n');print('Registered persistent mapping; no GPU work performed.')


if __name__=='__main__':main()
