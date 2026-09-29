# 文档检查记录

本机：Python 3.12.3，Sphinx 7.2.6。工作目录为 falconkv。

单页检查（exit 0，无警告）：

```bash
python3 -B -m sphinx -b html -C -W -n \
  design/p3/results/config-20260928/sphinx-smoke-source \
  design/p3/results/native-20260928/sphinx-page
```

该 source 的 index.rst 直接 include 当前 `LMCache/docs/source/getting_started/ascend_p3.rst`，不是历史内容快照。输出 `build succeeded.`。已检查 HTML 中 Native ownership、Build and development、Serving configuration migration、Validation boundary 章节及代码块。

全站检查（exit 2，在加载既有配置时失败）：

```bash
python3 -B -m sphinx -b html -W -n \
  p1-repos/LMCache/docs/source \
  design/p3/results/native-20260928/sphinx-full
```

报错关键部分：

```text
Running Sphinx v7.2.6
Configuration error:
File "p1-repos/LMCache/docs/source/conf.py", line 18, in <module>
    from sphinxawesome_theme import ThemeOptions
ModuleNotFoundError: No module named 'sphinxawesome_theme'
```

没有将全站结果合并成 host 通过项。后续准备完整文档依赖的环境仍需重跑全站构建。
