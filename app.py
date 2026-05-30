import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ==================== 配置 ====================
AMAP_KEY = "489518be45d4ca58a9bcb2e9ad39cf56"   # ←←← 必须修改

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def search_poi_amap(keywords, city=None, location=None, radius=3000):
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": AMAP_KEY,
        "keywords": keywords,
        "output": "JSON",
        "offset": 25,
        "page": 1,
        "extensions": "all"
    }
    if city:
        params["city"] = city
    if location:
        params["location"] = location
        params["radius"] = radius

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return data.get("pois", []) if data.get("status") == "1" else []
    except Exception as e:
        st.error(f"搜索出错: {e}")
        return []

def find_meaningful_clusters(all_pois, eps_km=2.0):
    if len(all_pois) < 2:
        return []

    coords, info = [], []
    for p in all_pois:
        try:
            lng, lat = map(float, p["location"].split(","))
            coords.append([lat, lng])
            info.append({
                "name": p.get("name", ""),
                "address": p.get("address", ""),
                "keyword": p.get("_keyword", ""),
                "lat": lat, "lng": lng
            })
        except:
            continue

    clustering = DBSCAN(eps=eps_km/6371, min_samples=2).fit(np.radians(coords))
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label != -1:
            clusters.setdefault(label, []).append(info[i])

    result = []
    for items in clusters.values():
        unique_kws = set(item["keyword"] for item in items)
        if len(unique_kws) >= 2:
            center_lat = np.mean([i["lat"] for i in items])
            center_lng = np.mean([i["lng"] for i in items])
            result.append({
                "center": (round(center_lat, 5), round(center_lng, 5)),
                "pois": items,
                "keywords": list(unique_kws),
                "count": len(items)
            })
    return result

# ==================== 初始化 session_state ====================
if "search_done" not in st.session_state:
    st.session_state.search_done = False
    st.session_state.results_by_keyword = {}
    st.session_state.clusters = []
    st.session_state.all_pois = []

# ==================== 界面 ====================
st.set_page_config(page_title="多POI商圈查找原型", layout="wide")
st.title("🗺️ 多POI商圈查找原型")
st.caption("输入多个关键词，查找同时存在这些店的区域（测试版）")

with st.sidebar:
    st.header("搜索设置")
    mode = st.radio("搜索模式", ["按城市搜索", "按坐标附近搜索"], horizontal=True)

    if mode == "按城市搜索":
        city = st.text_input("城市", value="深圳")
        location = None
    else:
        city = None
        lat = st.number_input("纬度", value=22.543, format="%.5f")
        lng = st.number_input("经度", value=114.057, format="%.5f")
        location = f"{lng},{lat}"
        radius = st.slider("搜索半径（米）", 1000, 5000, 3000, 500)

    keywords_text = st.text_area("关键词（每行一个）", 
                                  value="咖啡\n健身房\n超市", 
                                  height=100)
    eps_km = st.slider("商圈判断半径（公里）", 0.1, 4.0, 2.0, 0.1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("开始搜索", type="primary", use_container_width=True):
            keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]
            if keywords:
                all_pois = []
                results_by_keyword = {}
                for kw in keywords:
                    pois = search_poi_amap(kw, city=city, location=location, radius=radius if 'radius' in locals() else 3000)
                    for p in pois:
                        p["_keyword"] = kw
                    results_by_keyword[kw] = pois
                    all_pois.extend(pois)

                clusters = find_meaningful_clusters(all_pois, eps_km=eps_km)

                # 保存到 session_state
                st.session_state.results_by_keyword = results_by_keyword
                st.session_state.clusters = clusters
                st.session_state.all_pois = all_pois
                st.session_state.search_done = True
                st.rerun()   # 强制刷新，让结果持久显示
            else:
                st.warning("请输入关键词")

    with col2:
        if st.button("清空结果", use_container_width=True):
            st.session_state.search_done = False
            st.rerun()

# ==================== 显示结果（持久显示） ====================
if st.session_state.search_done:
    results_by_keyword = st.session_state.results_by_keyword
    clusters = st.session_state.clusters
    all_pois = st.session_state.all_pois
    keywords = list(results_by_keyword.keys())

    st.subheader("各关键词搜索结果")
    cols = st.columns(len(keywords))
    for i, (kw, pois) in enumerate(results_by_keyword.items()):
        with cols[i]:
            st.markdown(f"**{kw}** ({len(pois)} 条)")
            for p in pois[:6]:
                st.write(f"- {p.get('name')}")

    if clusters:
        st.subheader(f"🎯 发现 {len(clusters)} 个可能同时存在多个POI的区域")
        for idx, cluster in enumerate(clusters):
            with st.expander(f"区域 {idx+1} ｜ 包含：{', '.join(cluster['keywords'])}"):
                st.write(f"中心坐标：{cluster['center']}")
                for poi in cluster['pois']:
                    st.write(f"- [{poi['keyword']}] {poi['name']} ｜ {poi['address']}")

        # 地图
        st.subheader("地图可视化")
        center = [22.543, 114.057]
        if location:
            try:
                center = [float(lat), float(lng)]
            except:
                pass

        m = folium.Map(location=center, zoom_start=13)
        color_list = ["red", "blue", "green", "purple", "orange", "darkred"]
        color_map = {kw: color_list[i % len(color_list)] for i, kw in enumerate(keywords)}

        for poi in all_pois:
            try:
                lng_p, lat_p = map(float, poi["location"].split(","))
                folium.Marker(
                    [lat_p, lng_p],
                    popup=f"[{poi['_keyword']}] {poi.get('name')}<br>{poi.get('address','')}",
                    icon=folium.Icon(color=color_map.get(poi['_keyword'], "gray"))
                ).add_to(m)
            except:
                pass

        for cluster in clusters:
            folium.CircleMarker(
                location=cluster["center"],
                radius=18,
                color="black",
                fill=True,
                fill_color="yellow",
                fill_opacity=0.6,
                popup=f"可能商圈<br>{', '.join(cluster['keywords'])}"
            ).add_to(m)

        st_folium(m, width=1100, height=550)
    else:
        st.info("未找到同时包含多个关键词的密集区域，建议调整半径或换个位置。")

st.markdown("---")
st.caption("这是一个测试原型。数据来自高德地图。")
