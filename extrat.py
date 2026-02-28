import json
import os

def load_json_from_file(filename="ontology.json"):
    """
    从当前目录读取JSON文件
    """
    try:
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"错误: 在当前目录下未找到 {filename}")
            print(f"当前目录: {current_dir}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"成功从 {filename} 加载数据")
            return data
            
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式不正确 - {e}")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生错误 - {e}")
        return None

def extract_first_level_children(json_data):
    """
    提取微信小程序官方隐私接口和补充隐私数据类型的一级children
    """
    result = {}
    
    # 提取微信小程序官方隐私接口的一级children
    wechat_data = json_data["data_types_ontology"]["微信小程序官方隐私接口"]
    wechat_children = wechat_data["children"]
    
    wechat_result = {}
    for child_name, child_data in wechat_children.items():
        wechat_result[child_name] = {
            "id": child_data["id"],
            "terms": child_data["terms"]
        }
    
    # 提取补充隐私数据类型的一级children
    supplement_data = json_data["data_types_ontology"]["补充隐私数据类型"]
    supplement_children = supplement_data["children"]
    
    supplement_result = {}
    for child_name, child_data in supplement_children.items():
        supplement_result[child_name] = {
            "id": child_data["id"],
            "terms": child_data["terms"]
        }
        # 如果有二级children，也提取出来（可选）
        if "children" in child_data:
            supplement_result[child_name]["has_children"] = True
            supplement_result[child_name]["children_count"] = len(child_data["children"])
        else:
            supplement_result[child_name]["has_children"] = False
    
    result["微信小程序官方隐私接口_一级children"] = wechat_result
    result["补充隐私数据类型_一级children"] = supplement_result
    
    return result

def display_extracted_data(extracted_data):
    """
    格式化显示提取的数据
    """
    print("=" * 60)
    print("微信小程序官方隐私接口 - 一级children:")
    print("=" * 60)
    wechat_children = extracted_data["微信小程序官方隐私接口_一级children"]
    
    print(f"总数量: {len(wechat_children)}")
    print("\n详细信息:")
    for i, (name, info) in enumerate(wechat_children.items(), 1):
        print(f"{i}. {name}")
        print(f"   ID: {info['id']}")
        terms_preview = ', '.join(info['terms'][:3])
        if len(info['terms']) > 3:
            terms_preview += f" ...等{len(info['terms'])}个术语"
        print(f"   术语: {terms_preview}")
        print()
    
    print("\n" + "=" * 60)
    print("补充隐私数据类型 - 一级children:")
    print("=" * 60)
    supplement_children = extracted_data["补充隐私数据类型_一级children"]
    
    print(f"总数量: {len(supplement_children)}")
    print("\n详细信息:")
    for i, (name, info) in enumerate(supplement_children.items(), 1):
        print(f"{i}. {name}")
        print(f"   ID: {info['id']}")
        terms_preview = ', '.join(info['terms'][:3])
        if len(info['terms']) > 3:
            terms_preview += f" ...等{len(info['terms'])}个术语"
        print(f"   术语: {terms_preview}")
        if info.get('has_children'):
            print(f"   包含二级children: {info['children_count']}个")
        print()

def save_to_json(extracted_data, filename="extracted_children.json"):
    """
    将提取的数据保存为JSON文件
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {filepath}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def save_simple_list(extracted_data, filename="children_names.txt"):
    """
    保存简单的名称列表到文本文件
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("微信小程序官方隐私接口一级children名称列表\n")
            f.write("=" * 50 + "\n")
            
            wechat_names = list(extracted_data["微信小程序官方隐私接口_一级children"].keys())
            for i, name in enumerate(wechat_names, 1):
                f.write(f"{i}. {name}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("补充隐私数据类型一级children名称列表\n")
            f.write("=" * 50 + "\n")
            
            supplement_names = list(extracted_data["补充隐私数据类型_一级children"].keys())
            for i, name in enumerate(supplement_names, 1):
                f.write(f"{i}. {name}\n")
            
            # 添加统计信息
            f.write("\n" + "=" * 50 + "\n")
            f.write("统计信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"微信小程序官方隐私接口: {len(wechat_names)} 个\n")
            f.write(f"补充隐私数据类型: {len(supplement_names)} 个\n")
            f.write(f"总计: {len(wechat_names) + len(supplement_names)} 个\n")
        
        print(f"名称列表已保存到 {filepath}")
    except Exception as e:
        print(f"保存名称列表时出错: {e}")

# 主程序
if __name__ == "__main__":
    # 从文件加载数据
    print("正在从 ontology.json 文件读取数据...")
    data = load_json_from_file("ontology.json")
    
    if data is None:
        print("无法加载数据，程序退出。")
        print("请确保 ontology.json 文件存在于当前目录。")
        exit(1)
    
    # 检查数据结构
    if "data_types_ontology" not in data:
        print("错误: JSON文件中缺少 'data_types_ontology' 字段")
        exit(1)
    
    # 提取一级children
    extracted_data = extract_first_level_children(data)
    
    # 显示提取结果
    display_extracted_data(extracted_data)
    
    # 统计信息
    print("=" * 60)
    print("统计摘要:")
    print("=" * 60)
    
    wechat_count = len(extracted_data["微信小程序官方隐私接口_一级children"])
    supplement_count = len(extracted_data["补充隐私数据类型_一级children"])
    
    print(f"微信小程序官方隐私接口一级children数量: {wechat_count}")
    print(f"补充隐私数据类型一级children数量: {supplement_count}")
    print(f"总计: {wechat_count + supplement_count}")
    
    # 获取所有一级children的名称列表（控制台输出）
    print("\n" + "=" * 60)
    print("所有一级children名称:")
    print("=" * 60)
    
    print("\n1. 微信小程序官方隐私接口:")
    print("-" * 30)
    wechat_names = list(extracted_data["微信小程序官方隐私接口_一级children"].keys())
    for i, name in enumerate(wechat_names, 1):
        print(f"{i:2d}. {name}")
    
    print("\n2. 补充隐私数据类型:")
    print("-" * 30)
    supplement_names = list(extracted_data["补充隐私数据类型_一级children"].keys())
    for i, name in enumerate(supplement_names, 1):
        print(f"{i:2d}. {name}")
    
    # 保存到JSON文件
    save_to_json(extracted_data, "extracted_children.json")
    
    # 保存到文本文件（纯名称列表）
    save_simple_list(extracted_data, "children_names.txt")
    
    # 如果需要csv格式
    print("\n" + "=" * 60)
    print("CSV格式:")
    print("=" * 60)
    
    print("\n微信小程序官方隐私接口一级children:")
    print("序号,名称,ID")
    for i, (name, info) in enumerate(extracted_data["微信小程序官方隐私接口_一级children"].items(), 1):
        print(f'{i},"{name}","{info["id"]}"')
    
    print("\n补充隐私数据类型一级children:")
    print("序号,名称,ID")
    for i, (name, info) in enumerate(extracted_data["补充隐私数据类型_一级children"].items(), 1):
        has_children = "是" if info.get('has_children') else "否"
        print(f'{i},"{name}","{info["id"]}"')