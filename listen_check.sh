#!/usr/bin/env bash
# Play each flagged clip, show its reference text, record whether the audio matches it.
# Answers go to listen.txt (y = matches, n = different speech, p = partly).
cd "$(dirname "$0")"
IDS="118638_nJfFvNUB8tQkaGgp_0059 656852_dNHXkqqVPqWYhzis_0064
598753_406yMKxIdSDHRf8H_0085 146881_TkzR0GrQyK0vSTuj_0064 921151_kvxzesc5pGxL6dnk_0038
118638_nJfFvNUB8tQkaGgp_0005 118638_nJfFvNUB8tQkaGgp_0028 598753_406yMKxIdSDHRf8H_0112
656852_dNHXkqqVPqWYhzis_0038 598753_406yMKxIdSDHRf8H_0068"
for id in $IDS; do
  echo; echo "== $id"
  python3 -c "import json,sys;[print('REF:',r['text']) for r in map(json.loads,open('data/eval_300.jsonl')) if r['id']==sys.argv[1]]" "$id"
  while true; do
    aplay -q "data/test_segments/$id.wav"
    read -p "Does the audio match REF? [y/n/p, r=replay]: " a
    [ "$a" != "r" ] && break
  done
  echo "$id $a" >> listen.txt
done
echo; echo "saved to listen.txt:"; cat listen.txt
