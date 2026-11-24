"""Verify the saved final XLSX against the baseline and verified JSON records.

Read-only; uses the Python standard library, not an Excel writer.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

NS={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def col(i):
    result=''
    while i>=0:
        result=chr(65+i%26)+result
        i=i//26-1
    return result

def parse(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=[''.join(e.itertext()) for e in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        relations={e.attrib['Id']:e.attrib['Target'] for e in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        sheets={}
        for element in ET.fromstring(archive.read('xl/workbook.xml')).findall('x:sheets/x:sheet',NS):
            target=relations[element.attrib['{'+NS['r']+'}id']]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            cells={}; formulas={}
            xml=ET.fromstring(archive.read(target))
            for cell in xml.findall('x:sheetData/x:row/x:c',NS):
                address=cell.attrib['r'];kind=cell.attrib.get('t');value=cell.findtext('x:v',namespaces=NS)
                assert kind!='e',(element.attrib['name'],address,value)
                if kind=='s':value=strings[int(value)]
                elif kind=='inlineStr':value=''.join(cell.find('x:is',NS).itertext())
                elif kind=='b':value=value=='1'
                elif kind=='str':pass
                elif value is not None:
                    try:value=float(value)
                    except ValueError:pass
                cells[address]=value
                formula=cell.findtext('x:f',namespaces=NS)
                if formula is not None:formulas[address]=formula
            sheets[element.attrib['name']]={'cells':cells,'formulas':formulas,'xml':xml}
        tables={ET.fromstring(archive.read(n)).attrib['name']:ET.fromstring(archive.read(n)) for n in archive.namelist() if n.startswith('xl/tables/table') and n.endswith('.xml')}
        charts=[n for n in archive.namelist() if '/charts/chart' in n and n.endswith('.xml')]
        return sheets,tables,charts

def same(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b

def check(target,baseline,data_file,layout_file):
    old,old_tables,old_charts=parse(baseline)
    new,new_tables,new_charts=parse(target)
    assert list(old)==list(new) and len(new)==11
    assert set(old_tables)==set(new_tables) and len(old_charts)==len(new_charts)==1
    unchanged=('Held-out runs','Validation grid','Frozen recipes','OOF by seed','Per-fold summary','Five-fold summary','Job status','Failures')
    for name in unchanged:
        for address,value in old[name]['cells'].items():
            assert same(value,new[name]['cells'].get(address)),('Changed historical value',name,address,value,new[name]['cells'].get(address))
        for address,formula in old[name]['formulas'].items():
            assert formula==new[name]['formulas'].get(address),('Changed historical formula',name,address)
    for name,table in old_tables.items():
        assert table.find('x:tableStyleInfo',NS).attrib==new_tables[name].find('x:tableStyleInfo',NS).attrib
    data=json.loads(gzip.decompress(Path(data_file).read_bytes()))
    layout=json.loads(Path(layout_file).read_text())
    for name,section in [('Held-out runs','held_out'),('Validation grid','validation'),('Frozen recipes','frozen')]:
        keys=layout['layouts'][name]['keys'];start=layout['initial'][name]+2
        for row,record in enumerate(data[section],start):
            for i,key in enumerate(keys):
                if key in ('effective_gat_heads','selection_family') and section=='frozen':continue
                value=record.get(key,record.get('spec',{}).get(key))
                if isinstance(value,(list,dict)):value=json.dumps(value,separators=(',',':'))
                assert same(value,new[name]['cells'].get(col(i)+str(row))),('Source mismatch',name,row,key,value,new[name]['cells'].get(col(i)+str(row)))
    for i,record in enumerate(data['oof'],layout['initial']['OOF by seed']+2):
        for c,key in enumerate(layout['okeys']):
            assert same(record.get(key),new['OOF by seed']['cells'].get(col(c)+str(i))),('OOF mismatch',i,key)
    for i,record in enumerate(data['summaries'],layout['initial']['Five-fold summary']+2):
        for metric,values in record['metrics'].items():
            for suffix,key in [('mean','mean'),('seed_sd','seed_sd')]:
                address=col(layout['skeys'].index(metric+'_'+suffix))+str(i)
                assert same(values[key],new['Five-fold summary']['cells'][address]),('Summary mismatch',address)
    summary=new['Summary']['cells']
    assert summary['B5']==744 and summary['B6']==3139 and summary['G18']==0
    for row in range(11,16):
        for c in range(8):assert same(old['Summary']['cells'].get(col(c)+str(row)),summary.get(col(c)+str(row+37)))
    assert layout['layouts']['Held-out runs']['rows']==747
    assert layout['layouts']['Validation grid']['rows']==3139
    assert layout['layouts']['Frozen recipes']['rows']==40
    for name in ('Held-out runs','Validation grid','Frozen recipes'):
        pane=new[name]['xml'].find('x:sheetViews/x:sheetView/x:pane',NS)
        assert pane.attrib['xSplit']=='2' and pane.attrib['ySplit']=='1'
    return {'sheets':11,'paper_held_out_results':744,'validation_stage_entries':3139,'selected_recipes':40,
            'attention_confirmations':60,'attention_screens':594,'attention_screen_failures':6,
            'historical_scores':'all preserved','source_matches':True,'formula_errors':0}

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook',type=Path)
    parser.add_argument('--baseline',type=Path,default=here/'assets/baseline.xlsx')
    parser.add_argument('--data',type=Path,default=here/'assets/attention_records.json.gz')
    parser.add_argument('--layout',type=Path,required=True,help='layout.json written beside the exported workbook')
    args=parser.parse_args()
    print(json.dumps(check(args.workbook,args.baseline,args.data,args.layout),indent=2))
