import marimo as mo

def md_dump_parquet(data, labels=None, n=20, dataname='Data', section_flag=False):

    _text = ""

    if labels is not None:
        data = data.select(labels)
        _text  = "- 選択した列: "
        _text += ", ".join(labels)

    _schema_str = data._jdf.schema().treeString()
    _data_str = data._jdf.showString(n, 0, False) 
    
    if section_flag:
        return mo.vstack(
            [
                mo.md(
                    f"""
                    ### {dataname}: `parquet`の構造

                    {_text}
                    """
                ),
                mo.md(f"```text\n{_schema_str}\n```"),
                mo.md(
                    f"""
                    ### {dataname}: データデモ
                    """
                ),
                mo.md(f"```text\n{_data_str}\n```")
            ]
        )
    
    else:
        return mo.vstack(
            [
                mo.md(f"```text\n{_schema_str}\n```"),
                mo.md(f"```text\n{_data_str}\n```")
            ]
        )