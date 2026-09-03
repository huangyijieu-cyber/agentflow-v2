#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试样例：复现 Ray 分布式环境下 pyserini/JVM 类路径缺失问题

问题：单独运行 Python 脚本正常，但用 @ray.remote 启动后报：
    java.lang.NoClassDefFoundError: org/apache/lucene/index/SegmentInfos

原理：Ray 的远程进程不会自动继承主进程的环境变量（如 JAVA_HOME、CLASSPATH），
      导致 jnius 启动的 JVM 找不到 pyserini 依赖的 Lucene jar 包。
"""

import os
import sys
import tempfile
import shutil

os.environ['CLASSPATH'] = '/root/miniconda3/envs/agent_flow/lib/python3.10/site-packages/pyserini/resources/jars/anserini-1.1.1-fatjar.jar'

# ============================================================
# 步骤 0：准备一个小型 Lucene 索引（用于测试）
# ============================================================

def prepare_test_index(index_dir):
    """创建一个极简的 Lucene 索引，供 LuceneSearcher 加载"""
    try:
        from pyserini.index.lucene import LuceneIndexWriter
        from pyserini.encode import JsonlCollectionIterator

        # 准备测试数据
        docs = [
            {"id": "doc1", "contents": "hello world"},
            {"id": "doc2", "contents": "ray distributed test"},
        ]
        jsonl_path = os.path.join(index_dir, "docs.jsonl")
        with open(jsonl_path, "w") as f:
            for doc in docs:
                f.write(f'{{"id": "{doc["id"]}", "contents": "{doc["contents"]}"}}\n')

        # 写入索引
        writer = LuceneIndexWriter(index_dir)
        writer.write(jsonl_path)
        writer.close()
        return True
    except Exception as e:
        print(f"[WARN] 无法创建测试索引（可能缺少 pyserini.index）: {e}")
        print("[INFO] 将尝试直接加载现有索引...")
        return False


# ============================================================
# 步骤 1：本地模式测试（直接运行，预期成功）
# ============================================================

def test_local(index_dir):
    """本地直接运行：应该能正常加载 LuceneSearcher"""
    print("\n" + "="*60)
    print("[测试 1/3] 本地模式（直接运行，预期成功）")
    print("="*60)

    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_dir)
        print(f"[✓] 本地模式成功！Searcher 对象: {searcher}")
        return True
    except Exception as e:
        print(f"[✗] 本地模式失败: {e}")
        return False


# ============================================================
# 步骤 2：Ray 远程模式（不传递环境变量，预期复现错误）
# ============================================================

def test_ray_remote_without_env(index_dir):
    """Ray 远程模式，不传递环境变量：预期复现 NoClassDefFoundError"""
    import ray

    print("\n" + "="*60)
    print("[测试 2/3] Ray 远程模式（不传递 JAVA/CLASSPATH，预期失败）")
    print("="*60)

    ray.init(ignore_reinit_error=True)

    @ray.remote
    def remote_load_searcher(index_dir):
        # 注意：这里不设置任何环境变量
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_dir)
        return f"Remote OK: {searcher}"

    try:
        result = ray.get(remote_load_searcher.remote(index_dir))
        print(f"[✓] 远程模式成功: {result}")
        return True
    except Exception as e:
        print(f"[✗] 远程模式失败（已复现问题）:")
        # 打印关键错误信息
        error_msg = str(e)
        if "NoClassDefFoundError" in error_msg:
            print(f"    → 命中目标错误: NoClassDefFoundError")
            print(f"    → 原因: Ray 远程进程中 JVM 找不到 Lucene 类")
        print(f"    详细错误: {error_msg[:500]}")
        return False
    finally:
        ray.shutdown()


# ============================================================
# 步骤 3：Ray 远程模式（正确传递环境变量，预期成功）
# ============================================================

def test_ray_remote_with_env(index_dir, java_home, classpath):
    """Ray 远程模式，正确传递环境变量：预期成功"""
    import ray

    print("\n" + "="*60)
    print("[测试 3/3] Ray 远程模式（传递 JAVA/CLASSPATH，预期成功）")
    print("="*60)

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "JAVA_HOME": java_home,
                "CLASSPATH": classpath,
                "PATH": f"{java_home}/bin:{os.environ.get('PATH', '')}",
            }
        }
    )

    @ray.remote
    def remote_load_searcher(index_dir):
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(index_dir)
        return f"Remote OK: {searcher}"

    try:
        result = ray.get(remote_load_searcher.remote(index_dir))
        print(f"[✓] 远程模式成功: {result}")
        return True
    except Exception as e:
        print(f"[✗] 远程模式失败: {e}")
        return False
    finally:
        ray.shutdown()


# ============================================================
# 诊断工具：打印环境信息
# ============================================================

def diagnose_env():
    """打印当前环境的关键信息，帮助排查"""
    print("\n" + "="*60)
    print("[环境诊断]")
    print("="*60)

    print(f"PYTHON: {sys.executable}")
    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', 'NOT SET')}")
    print(f"CLASSPATH: {os.environ.get('CLASSPATH', 'NOT SET')}")

    # 尝试找到 anserini jar
    import subprocess
    try:
        result = subprocess.run(
            ['find', os.path.dirname(sys.executable), '-name', '*anserini*.jar'],
            capture_output=True, text=True, timeout=10
        )
        jars = [l for l in result.stdout.strip().split('\n') if l]
        print(f"找到 {len(jars)} 个 anserini jar:")
        for j in jars[:5]:
            print(f"  - {j}")
    except Exception as e:
        print(f"查找 jar 失败: {e}")

    # 检查 jnius 状态
    try:
        import jnius
        print(f"jnius 版本: {jnius.__version__ if hasattr(jnius, '__version__') else 'unknown'}")
    except ImportError:
        print("jnius 未安装")


# ============================================================
# 主入口
# ============================================================

def main():
    # 0. 诊断环境
    diagnose_env()

    # 1. 准备临时索引目录
    index_dir = tempfile.mkdtemp(prefix="test_lucene_index_")
    print(f"\n[INFO] 测试索引目录: {index_dir}")

    # 尝试创建索引，如果失败就用一个空目录（会报不同错误，但也能测试类加载）
    prepare_test_index(index_dir)

    # 2. 运行三组测试
    local_ok = test_local(index_dir)
    ray_fail_ok = test_ray_remote_without_env(index_dir)

    # 3. 如果本地成功，尝试修复后的 Ray 模式
    java_home = os.environ.get('JAVA_HOME', '')
    classpath = os.environ.get('CLASSPATH', '')

    if local_ok and java_home and classpath:
        ray_fix_ok = test_ray_remote_with_env(index_dir, java_home, classpath)
    else:
        print("\n[SKIP] 测试 3: 本地环境变量不完整，跳过修复测试")
        print("       请确保 JAVA_HOME 和 CLASSPATH 已设置，再运行测试 3")
        ray_fix_ok = None

    # 4. 总结
    print("\n" + "="*60)
    print("[测试结果总结]")
    print("="*60)
    print(f"  本地模式:       {'PASS' if local_ok else 'FAIL'}")
    print(f"  Ray 无环境变量: {'FAIL (已复现)' if not ray_fail_ok else 'PASS (未复现)'}")
    if ray_fix_ok is not None:
        print(f"  Ray 有环境变量: {'PASS (修复成功)' if ray_fix_ok else 'FAIL'}")

    # 清理
    shutil.rmtree(index_dir, ignore_errors=True)
    print(f"\n[INFO] 已清理临时目录: {index_dir}")


if __name__ == "__main__":
    main()