import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import ee
from openai import OpenAI
from geopy.geocoders import GoogleV3
import json
import folium
from streamlit_folium import st_folium
import time

# --- 환경 변수 + GEE 초기화 함수 (최초 1회만 실행) ---
@st.cache_resource
def init_resources():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    if not OPENAI_API_KEY or not GOOGLE_MAPS_API_KEY:
        st.error("환경변수 OPENAI_API_KEY 또는 GOOGLE_MAPS_API_KEY가 설정되어 있지 않습니다.")
        st.stop()

    client = OpenAI(api_key=OPENAI_API_KEY)
    geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY)

    config_path = os.path.abspath('../../Credentials/config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    service_account = config['service_account']
    key_path = os.path.abspath(config['key_path'])

    credentials = ee.ServiceAccountCredentials(service_account, key_path)
    ee.Initialize(credentials)

    return client, geolocator

# 최초 1회만 초기화
client, geolocator = init_resources()

st.title("NDVI, EVI, SAVI, GCI 변화 분석 및 상세 리포트 생성")

# --- 세션 상태 초기화 ---
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# --- 사용자 입력 ---
user_input = st.text_input(
    "분석 요청 예시: show me the NDVI changes of last 2 years 10 miles around at 3711 sunnyside ct. rochester hills, mi 48306",
    value=st.session_state.user_input
)

# --- 실행 버튼 ---
if st.button("분석 시작"):
    if user_input.strip() == "":
        st.warning("분석 요청 내용을 입력해주세요.")
    else:
        st.session_state.run_analysis = True
        st.session_state.user_input = user_input

def run_full_analysis(user_input):
    try:
        total_start = time.time()

        # 1) OpenAI에 주소, 거리, 기간 파싱 요청
        step1_placeholder = st.empty()
        step1_placeholder.markdown("⚙️ ** OpenAI 기반 입력내용 수집 중...**")
        step1_start = time.time()

        parse_prompt = f"""
        Extract these info from the text below:
        - The address as a single string
        - The radius in miles as a number
        - The years count (number of past years to analyze)

        Text:
        \"\"\"{user_input}\"\"\" 

        Reply ONLY as a JSON object with keys: "address", "radius_miles", "years".
        For example: {{ "address": "3711 Sunnyside Ct, Rochester Hills, MI 48306", "radius_miles": 10, "years": 2 }}
        """

        parse_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": parse_prompt}]
        )
        parsed = parse_response.choices[0].message.content
        parsed_json = json.loads(parsed)
        step1_end = time.time()
        step1_placeholder.markdown(f"✅ OpenAI 기반 입력내용 수집 완료  (소요시간: {step1_end - step1_start:.2f}초)")

        address = parsed_json["address"]
        radius_miles = float(parsed_json["radius_miles"])
        years_num = int(parsed_json["years"])

        st.markdown(f"**주소:** {address}")
        st.markdown(f"**반경(miles):** {radius_miles}")
        st.markdown(f"**분석 기간(년):** {years_num}")

        # 2) 주소 → 위도,경도 변환
        step2_placeholder = st.empty()
        step2_placeholder.markdown("⚙️ ** Google Map 기반 주소를 위도, 경도로 변환 중...**")
        step2_start = time.time()

        location = geolocator.geocode(address)
        if location is None:
            step2_placeholder.empty()
            st.error("주소를 위치로 변환할 수 없습니다.")
            return

        lat, lng = location.latitude, location.longitude
        step2_end = time.time()
        step2_placeholder.markdown(f"✅ Google Map 기반 주소를 위도, 경도로 변환 완료 (소요시간: {step2_end - step2_start:.2f}초)")

        st.markdown(f"**위도:** {lat:.6f}, **경도:** {lng:.6f}")

        # 2.1) Folium 지도 생성 및 반경 원 표시
        radius_meters = radius_miles * 1609.34
        m = folium.Map(location=[lat, lng], zoom_start=12)

        folium.CircleMarker(
            location=[lat, lng],
            radius=5,
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.7,
            popup=address
        ).add_to(m)

        folium.Circle(
            location=[lat, lng],
            radius=radius_meters,
            color='purple',
            fill=True,
            fill_opacity=0.2,
            popup=f"{radius_miles} miles radius"
        ).add_to(m)

        from geopy.distance import distance
        sw = distance(meters=radius_meters).destination((lat, lng), bearing=225)
        ne = distance(meters=radius_meters).destination((lat, lng), bearing=45)
        m.fit_bounds([[sw.latitude, sw.longitude], [ne.latitude, ne.longitude]])

        st.subheader("요청 위치 및 반경 지도 (전체 영역이 보이도록 자동 조절)")
        st_folium(m, width=700, height=500)

        # 3) GEE 영역 생성
        step3_placeholder = st.empty()
        step3_placeholder.markdown("⚙️ ** GEE 영역 생성 중...**")
        step3_start = time.time()

        geometry = ee.Geometry.Point([lng, lat]).buffer(radius_meters)

        step3_end = time.time()
        step3_placeholder.markdown(f"✅ GEE 영역 생성 완료 (소요시간: {step3_end - step3_start:.2f}초)")

        # 4) 분석 연도 리스트 생성
        step4_placeholder = st.empty()
        step4_placeholder.markdown("⚙️ ** 분석할 연도 리스트 생성 중...**")
        step4_start = time.time()

        current_year = datetime.datetime.now().year
        years = list(range(current_year - years_num, current_year))

        step4_end = time.time()
        step4_placeholder.markdown(f"✅ 분석할 연도 리스트 생성 완료 (소요시간: {step4_end - step4_start:.2f}초)")

        # --- 지수 계산 함수 ---
        def get_yearly_indices(year):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                          .filterDate(start_date, end_date)
                          .filterBounds(geometry)
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                          .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11']))

            count = collection.size().getInfo()
            if count == 0:
                return None

            image = collection.median()

            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
            ).rename('EVI')
            savi = image.expression(
                '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
                {'NIR': image.select('B8'), 'RED': image.select('B4')}
            ).rename('SAVI')
            gci = image.expression(
                '(NIR / GREEN) - 1',
                {'NIR': image.select('B8'), 'GREEN': image.select('B3')}
            ).rename('GCI')

            indices_img = ndvi.addBands([evi, savi, gci])
            return indices_img

        def get_mean_index(img, band_name):
            if img is None:
                return np.nan
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            )
            val = stats.get(band_name).getInfo()
            return val if val is not None else np.nan

        # 5) 연도별 지수 계산
        step5_placeholder = st.empty()
        step5_placeholder.markdown("⚙️ ** 연도별 지수 계산 중...**")
        step5_start = time.time()

        mean_indices_values = []
        indices_images = {}
        for y in years:
            indices_img = get_yearly_indices(y)
            ndvi_val = get_mean_index(indices_img, 'NDVI')
            evi_val = get_mean_index(indices_img, 'EVI')
            savi_val = get_mean_index(indices_img, 'SAVI')
            gci_val = get_mean_index(indices_img, 'GCI')

            mean_indices_values.append({
                'year': y,
                'NDVI': ndvi_val,
                'EVI': evi_val,
                'SAVI': savi_val,
                'GCI': gci_val
            })
            indices_images[y] = indices_img

        df_indices = pd.DataFrame(mean_indices_values).sort_values('year')

        step5_end = time.time()
        step5_placeholder.markdown(f"✅ 연도별 지수 계산 완료 (소요시간: {step5_end - step5_start:.2f}초)")

        st.subheader("연도별 평균 지수 값 (NDVI, EVI, SAVI, GCI)")
        st.dataframe(df_indices)

        ndvi_values = df_indices['NDVI'].fillna(0).tolist()
        change_desc = []
        for i in range(1, len(ndvi_values)):
            diff = ndvi_values[i] - ndvi_values[i-1]
            if diff > 0.01:
                change_desc.append(f"{df_indices.iloc[i]['year']}년에 NDVI가 약 {diff:.3f}만큼 증가하여 식생이 개선된 것으로 보입니다.")
            elif diff < -0.01:
                change_desc.append(f"{df_indices.iloc[i]['year']}년에 NDVI가 약 {abs(diff):.3f}만큼 감소하여 식생이 악화된 것으로 추정됩니다.")
            else:
                change_desc.append(f"{df_indices.iloc[i]['year']}년에는 NDVI가 거의 변화하지 않았습니다.")
        change_summary = "\n".join(change_desc) if change_desc else "NDVI 변화 데이터가 충분하지 않아 추세 분석이 어렵습니다."

        indices_summary_lines = []
        for _, row in df_indices.iterrows():
            line = (
                f"- {row['year']}: "
                f"NDVI = {row['NDVI']:.3f}, "
                f"EVI = {row['EVI']:.3f}, "
                f"SAVI = {row['SAVI']:.3f}, "
                f"GCI = {row['GCI']:.3f}"
                if not np.isnan(row['NDVI']) else
                f"- {row['year']}: 데이터 없음"
            )
            indices_summary_lines.append(line)
        indices_summary = "\n".join(indices_summary_lines)

        # 6) OpenAI 리포트 생성
        step6_placeholder = st.empty()
        step6_placeholder.markdown("⚙️ ** OpenAI에서 상세 리포트 생성 중...**")
        step6_start = time.time()

        report_prompt = f"""
        당신은 환경 과학 전문가입니다.
        아래는 위도 {lat:.6f}, 경도 {lng:.6f} 위치 주변 {radius_miles}마일 반경 내 지난 {years_num}년간의 연도별 평균 NDVI, EVI, SAVI, GCI 지수 값입니다:

        {indices_summary}

        위 데이터를 기반으로,
        1) 각 지수(NDVI, EVI, SAVI, GCI)가 무엇을 의미하는지 비전문가도 이해할 수 있게 간략히 설명하세요.
        2) 연도별 지수 변화 추세를 분석하고, 그 의미를 구체적으로 해석하세요.
        3) 지수들의 증가 혹은 감소 원인이 될 수 있는 환경적 요인이나 식생 건강 상태를 예측하여 설명하세요.
        4) 해당 지역의 식생 상태에 관한 종합적인 평가와, 필요하다면 권장 조치(예: 모니터링 강화, 보호, 복원 등)를 제시하세요.

        NDVI 변화 추이에 대한 구체적인 분석은 아래와 같습니다:
        {change_summary}

        위 내용을 한국어로 명확하고 자세하게 작성해 주세요.
        """
        report_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert environmental scientist."},
                {"role": "user", "content": report_prompt}
            ],
            max_tokens=1500
        )
        report = report_response.choices[0].message.content
        step6_end = time.time()
        step6_placeholder.markdown(f"✅ 상세 리포트 생성 완료 (소요시간: {step6_end - step6_start:.2f}초)")

        st.subheader("🌿 상세 환경 및 식생 변화 리포트")
        st.markdown(report)

        # --- NDVI 변화량 지도 생성 및 출력 ---
        def create_ndvi_change_map(ndvi_img_latest, ndvi_img_prev, center_lat, center_lon, radius_miles, address):
            if ndvi_img_latest is None or ndvi_img_prev is None:
                return None

            # NDVI 변화량 계산
            ndvi_change = ndvi_img_latest.select('NDVI').subtract(
                ndvi_img_prev.select('NDVI')
            ).rename('NDVI_change')

            # 지도 생성
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

            # NDVI 변화량 시각화 파라미터
            ndvi_change_params = {
                'min': -0.5,
                'max': 0.5,
                'palette': ['red', 'white', 'green']
            }
            map_id_dict = ndvi_change.getMapId(ndvi_change_params)

            # 타일 레이어 추가
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='NDVI Change',
                overlay=True,
                control=True
            ).add_to(m)

            # 보라색 반경 원 및 위치 점 추가
            radius_meters = radius_miles * 1609.34

            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=5,
                color='purple',
                fill=True,
                fill_color='purple',
                fill_opacity=0.7,
                popup=address
            ).add_to(m)

            folium.Circle(
                location=[center_lat, center_lon],
                radius=radius_meters,
                color='purple',
                fill=True,
                fill_opacity=0.2,
                popup=f"{radius_miles} miles radius"
            ).add_to(m)

            # 반경 원 전체가 보이도록 지도 뷰 조절
            from geopy.distance import distance
            sw = distance(meters=radius_meters).destination((center_lat, center_lon), bearing=225)
            ne = distance(meters=radius_meters).destination((center_lat, center_lon), bearing=45)
            m.fit_bounds([[sw.latitude, sw.longitude], [ne.latitude, ne.longitude]])

            # 레이어 컨트롤 추가
            folium.LayerControl().add_to(m)

            return m

        # --- NDVI 변화량 지도 출력 ---
        if len(years) >= 2:
            st.subheader("NDVI 변화량 지도 (최근 2년)")

            if indices_images.get(years[-1]) is None or indices_images.get(years[-2]) is None:
                st.info("최근 2년간 Sentinel-2 위성 데이터가 부족하여 NDVI 변화량 지도를 생성할 수 없습니다.")
            else:
                ndvi_change_map = create_ndvi_change_map(
                    indices_images[years[-1]],
                    indices_images[years[-2]],
                    lat,
                    lng,
                    radius_miles,
                    address
                )
                st_folium(ndvi_change_map, width=700, height=500)
        else:
            st.info("NDVI 변화량 지도는 최소 2년 이상 데이터를 선택해야 생성됩니다.")


        total_end = time.time()
        st.markdown(f"### 전체 분석 소요 시간: {(total_end - total_start):.2f}초")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")

# --- 실행 ---
if st.session_state.run_analysis:
    run_full_analysis(st.session_state.user_input)
