## wavファイルに変換する方法
```
for f in ./data/pre-process/*.m4a; do ffmpeg -i "$f" "${f%.m4a}.wav"; done

```