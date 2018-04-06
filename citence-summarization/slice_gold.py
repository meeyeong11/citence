# -*-coding:utf-8-*-

from util.abs_paths import get_paths
from util.file_op import read_file_utf8, write_file_utf8

paths = get_paths('./data')
for fid in range(len(paths)):
    print '[',fid,']', paths[fid]
    lines,_, summary = read_file_utf8(paths[fid]).split(u'---')
    write_file_utf8(paths[fid].replace('data','gold_summary'), summary.replace('\n',''))