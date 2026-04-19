import streamlit as st
import pandas as pd
from datetime import datetime
import os
import cv2
from PIL import Image
from compare import DashboardComparator

RUSSIAN_TEXTS = {
    "app_title": "Система управления версиями дашбордов",
    "upload_header": "Загрузка новой версии",
    "dashboard_name": "Название дашборда",
    "screenshot": "Скриншот дашборда",
    "version_description": "Описание версии",
    "version_description_placeholder": "Что изменилось в этой версии?",
    "upload_button": "Загрузить",
    "upload_success": "Загружена версия",
    "changes_from_previous": "Отличий от предыдущей версии",
    "upload_first": "Загружена первая версия",
    "history_tab": "История версий",
    "compare_tab": "Сравнение дашбордов",
    "no_dashboards": "Нет загруженных дашбордов. Загрузите первый скриншот в боковой панели.",
    "select_dashboard": "Выберите дашборд",
    "version_label": "Версия",
    "description_label": "Описание",
    "no_description": "Нет описания",
    "changes_from_previous_label": "Изменения относительно предыдущей версии",
    "similarity": "Схожесть",
    "changed": "Изменено",
    "verdict": "Вердикт",
    "compare_header": "Сравнение двух версий",
    "first_dashboard": "Первый дашборд",
    "second_dashboard": "Второй дашборд",
    "compare_button": "Сравнить",
    "analysis": "Анализ",
    "visual_comparison": "Визуальное сравнение",
    "difference_overlay": "Наложение различий (красным отмечены изменения)",
    "detailed_heatmap": "Детальная тепловая карта"
}

STORAGE_DIR = "storage"
SCREENSHOTS_DIR = os.path.join(STORAGE_DIR, "screenshots")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.csv")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if not os.path.exists(METADATA_FILE):
    empty_dataframe = pd.DataFrame(columns=["dashboard_id", "version", "timestamp", "file_path", "description"])
    empty_dataframe.to_csv(METADATA_FILE, index=False)

def load_metadata():
    return pd.read_csv(METADATA_FILE)

def save_metadata(dataframe):
    dataframe.to_csv(METADATA_FILE, index=False)

def get_dashboard_list():
    dataframe = load_metadata()
    if dataframe.empty:
        return []
    return dataframe["dashboard_id"].unique().tolist()

def get_versions(dashboard_id):
    dataframe = load_metadata()
    versions_dataframe = dataframe[dataframe["dashboard_id"] == dashboard_id].sort_values("version")
    return versions_dataframe.to_dict("records")

st.set_page_config(page_title=RUSSIAN_TEXTS["app_title"], layout="wide")
st.title(RUSSIAN_TEXTS["app_title"])
st.markdown("---")

with st.sidebar:
    st.header(RUSSIAN_TEXTS["upload_header"])

    dashboard_name_input = st.text_input(RUSSIAN_TEXTS["dashboard_name"])
    uploaded_file = st.file_uploader(RUSSIAN_TEXTS["screenshot"], type=["png", "jpg", "jpeg"])
    version_description_input = st.text_area(
        RUSSIAN_TEXTS["version_description"],
        placeholder=RUSSIAN_TEXTS["version_description_placeholder"]
    )

    if st.button(RUSSIAN_TEXTS["upload_button"]) and dashboard_name_input and uploaded_file:
        timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_versions = get_versions(dashboard_name_input)
        new_version_number = len(existing_versions) + 1
        filename = f"{dashboard_name_input}_v{new_version_number}_{timestamp_string}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)

        with open(filepath, "wb") as destination_file:
            destination_file.write(uploaded_file.getbuffer())

        metadata_dataframe = load_metadata()
        new_row = {
            "dashboard_id": dashboard_name_input,
            "version": new_version_number,
            "timestamp": timestamp_string,
            "file_path": filepath,
            "description": version_description_input
        }
        metadata_dataframe = pd.concat([metadata_dataframe, pd.DataFrame([new_row])], ignore_index=True)
        save_metadata(metadata_dataframe)

        if len(existing_versions) > 0:
            previous_version = existing_versions[-1]
            comparator = DashboardComparator()
            comparison_result = comparator.compare(previous_version["file_path"], filepath)

            st.sidebar.success(f"{RUSSIAN_TEXTS['upload_success']} {new_version_number}")
            st.sidebar.info(f"{RUSSIAN_TEXTS['changes_from_previous']}: {comparison_result['change_percentage']}%")
        else:
            st.sidebar.success(RUSSIAN_TEXTS["upload_first"])

        st.rerun()

main_tab, compare_tab = st.tabs([RUSSIAN_TEXTS["history_tab"], RUSSIAN_TEXTS["compare_tab"]])

with main_tab:
    st.header(RUSSIAN_TEXTS["history_tab"])

    available_dashboards = get_dashboard_list()
    if not available_dashboards:
        st.info(RUSSIAN_TEXTS["no_dashboards"])
    else:
        selected_dashboard = st.selectbox(RUSSIAN_TEXTS["select_dashboard"], available_dashboards)
        dashboard_versions = get_versions(selected_dashboard)

        for version_data in reversed(dashboard_versions):
            expander_label = f"{RUSSIAN_TEXTS['version_label']} {version_data['version']} — {version_data['timestamp']}"
            with st.expander(expander_label):
                left_column, right_column = st.columns([1, 1])

                with left_column:
                    screenshot_image = Image.open(version_data["file_path"])
                    st.image(screenshot_image, caption=f"{RUSSIAN_TEXTS['version_label']} {version_data['version']}", use_container_width=True)

                with right_column:
                    description_text = version_data["description"] if version_data["description"] else RUSSIAN_TEXTS["no_description"]
                    st.markdown(f"**{RUSSIAN_TEXTS['description_label']}:** {description_text}")

                    if version_data["version"] > 1:
                        previous_version_data = dashboard_versions[version_data["version"] - 2]
                        comparator = DashboardComparator()
                        comparison_result = comparator.compare(
                            previous_version_data["file_path"],
                            version_data["file_path"]
                        )

                        st.markdown(f"**{RUSSIAN_TEXTS['changes_from_previous_label']}:**")
                        st.markdown(f"- {RUSSIAN_TEXTS['similarity']}: {comparison_result['similarity'] * 100:.1f}%")
                        st.markdown(f"- {RUSSIAN_TEXTS['changed']}: {comparison_result['change_percentage']:.1f}%")
                        st.markdown(f"- {RUSSIAN_TEXTS['verdict']}: {comparison_result['verdict']}")
                        st.markdown(f"- {comparison_result['description']}")

with compare_tab:
    st.header(RUSSIAN_TEXTS["compare_header"])

    available_dashboards = get_dashboard_list()
    if len(available_dashboards) < 1:
        st.info(RUSSIAN_TEXTS["no_dashboards"])
    else:
        left_column, right_column = st.columns(2)

        with left_column:
            first_dashboard = st.selectbox(RUSSIAN_TEXTS["first_dashboard"], available_dashboards, key="first_dashboard")
            first_versions = get_versions(first_dashboard)
            first_version_selection = st.selectbox(
                RUSSIAN_TEXTS["version_label"],
                [f"v{v['version']}" for v in first_versions],
                key="first_version"
            )
            first_version_index = int(first_version_selection.replace("v", "")) - 1
            first_image_path = first_versions[first_version_index]["file_path"]

        with right_column:
            second_dashboard = st.selectbox(RUSSIAN_TEXTS["second_dashboard"], available_dashboards, key="second_dashboard")
            second_versions = get_versions(second_dashboard)
            second_version_selection = st.selectbox(
                RUSSIAN_TEXTS["version_label"],
                [f"v{v['version']}" for v in second_versions],
                key="second_version"
            )
            second_version_index = int(second_version_selection.replace("v", "")) - 1
            second_image_path = second_versions[second_version_index]["file_path"]

        if st.button(RUSSIAN_TEXTS["compare_button"]):
            comparator = DashboardComparator()
            comparison_result = comparator.compare(first_image_path, second_image_path)

            st.markdown("---")

            metric_columns = st.columns(3)
            with metric_columns[0]:
                st.metric(RUSSIAN_TEXTS["similarity"], f"{comparison_result['similarity'] * 100:.1f}%")
            with metric_columns[1]:
                st.metric(RUSSIAN_TEXTS["changed"], f"{comparison_result['change_percentage']:.1f}%")
            with metric_columns[2]:
                st.metric(RUSSIAN_TEXTS["verdict"], comparison_result["verdict"])

            st.markdown(f"**{RUSSIAN_TEXTS['analysis']}:** {comparison_result['description']}")

            st.markdown("---")
            st.subheader(RUSSIAN_TEXTS["visual_comparison"])

            comparison_columns = st.columns(2)
            with comparison_columns[0]:
                st.image(first_image_path, caption=f"{first_dashboard} - {first_version_selection}", use_container_width=True)
            with comparison_columns[1]:
                st.image(second_image_path, caption=f"{second_dashboard} - {second_version_selection}", use_container_width=True)

            st.subheader(RUSSIAN_TEXTS["difference_overlay"])
            overlay_rgb = cv2.cvtColor(comparison_result['overlay'], cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, use_container_width=True)

            st.subheader(RUSSIAN_TEXTS["detailed_heatmap"])
            difference_map = comparison_result['difference_map']
            colored_heatmap = cv2.applyColorMap(difference_map, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, use_container_width=True)