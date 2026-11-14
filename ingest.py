import argparse
import sys

from config import COLLECTION_NAME, PDF_PATH, STRUCTURED_DIR
from campusmind.knowledge_base import build_knowledge_base, load_structured_documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CampusMind：校验结构化资料并建立知识库。")
    parser.add_argument(
        "--check-only", action="store_true",
        help="只检查源 PDF、OCR、切片和父子关联；不调用向量接口、不修改数据库",
    )
    parser.add_argument(
        "--collection-name", default=COLLECTION_NAME,
        help="正式入库的目标集合名称，默认使用 config.py 中的当前集合",
    )
    args = parser.parse_args(argv)
    try:
        print("===== CampusMind 资料入库 =====", flush=True)
        print(f"源资料：{PDF_PATH}")
        print(f"结构化资料目录：{STRUCTURED_DIR}", flush=True)
        if args.check_only:
            chunks, parents, manifest, _ = load_structured_documents(PDF_PATH)
            print(f"资料校验通过：{len(manifest['ocr_files'])} 页，"
                  f"{len(chunks)} 个子片段，{len(parents)} 份父级资料。")
            print("本次仅检查输入资料，未检查目标集合状态，也未执行入库。")
            return 0

        print(f"目标集合：{args.collection_name}", flush=True)
        report = build_knowledge_base(PDF_PATH, args.collection_name)
        print("\n===== 入库完成 =====")
        print(f"集合：{report['collection_name']}")
        print(f"资料页数：{report['page_count']}")
        print(f"子片段：{report['chunk_count']}；父级资料：{report['parent_count']}")
        print(f"本次新增：{report['added']}；已有并跳过：{report['skipped']}")
        print(f"已核对的入库片段：{report['stored_count']}")
        print(f"父级资料快照：{report['parent_snapshot']}")
        if args.collection_name != COLLECTION_NAME:
            print("若要使用本次集合问答，请更新 config.py 或 .env 中的当前集合设置。")
        print("请重启正在运行的问答服务，使其重新加载知识库。")
        return 0
    except FileNotFoundError as exc:
        print(f"\n缺少资料文件：{exc}", file=sys.stderr)
        print("请先准备 PDF、OCR 和结构化资料。运行 python -m campusmind.document_processing "
              "可处理已有 OCR 结果；该命令不会自动完成 OCR。", file=sys.stderr)
    except ValueError as exc:
        print(f"\n入库检查未通过：{exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n已中断。修复问题后可重新执行，同版本已成功入库的片段会被跳过。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n入库未完成（{type(exc).__name__}）。请检查网络、密钥、额度和目录权限后重试。",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
