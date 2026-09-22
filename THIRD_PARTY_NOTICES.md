# Third-party notices

LiLo-VLA redistributes the following third-party work. Each is MIT licensed;
the original copyright notices are reproduced in full below, as MIT requires.

| Path in this repo | Upstream project | Copyright |
|---|---|---|
| `prismatic/` (87 files, verbatim) | [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | Moo Jin Kim, Chelsea Finn, Percy Liang |
| `scripts/train/finetune.py` | [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | Moo Jin Kim, Chelsea Finn, Percy Liang |
| `configs/bddl/` (100 BDDL files) | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone |
| `configs/init_states/` (9 files) | derived from LIBERO scene definitions | as above |
| `third_party/rlds_dataset_builder/` | [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) | Karl Pertsch et al. |

`lilo_vla/libero_compat.py` reproduces five behavioural patches from LIBERO's
sources; the patched function bodies are copied verbatim and are covered by the
LIBERO notice below.

---

## OpenVLA-OFT

```
MIT License

Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## LIBERO

```
MIT License

Copyright (c) 2023 Lifelong Robot Learning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
