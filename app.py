import streamlit as st
import pandas as pd
from datetime import datetime
import os
import cv2
import shutil
from PIL import Image
from compare import DashboardComparator

RUSSIAN_TEXTS = {
    "app_title": "Система управления версиями дашбордов",
    "upload_header": "Загрузка новой версии",
    "existing_dashboard": "Выберите существующий дашборд",
    "or_create_new": "Или создайте новый",
    "new_dashboard_name": "Название нового дашборда",
    "select_dashboard_to_update": "Выберите дашборд для обновления",
    "screenshot": "Скриншот дашборда",
    "version_description": "Описание версии",
    "version_description_placeholder": "Что изменилось в этой версии?",
    "upload_button": "Загрузить",
    "upload_success": "Успешно загружена версия",
    "upload_success_message": "Дашборд '{name}' версия {version} загружена",
    "changes_from_previous": "Отличий от предыдущей версии",
    "upload_first": "Загружена первая версия",
    "history_tab": "История версий",
    "compare_tab": "Сравнение дашбордов",
    "no_dashboards": "Нет загруженных дашбордов. Загрузите первый скриншот.",
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
    "detailed_heatmap": "Детальная тепловая карта",
    "upload_success_title": "Загрузка выполнена",
    "version_info": "Версия {version} дашборда '{name}'",
    "close": "Закрыть",
    "new_version_added": "Добавлена новая версия",
    "delete_dashboard": "Удалить дашборд",
    "delete_version": "Удалить эту версию",
    "confirm_delete_dashboard": "Вы уверены, что хотите удалить дашборд '{name}'? Все версии будут удалены без возможности восстановления.",
    "confirm_delete_version": "Вы уверены, что хотите удалить версию {version} дашборда '{name}'?",
    "yes_delete": "Да, удалить",
    "cancel": "Отмена",
    "deleted_success": "Удаление выполнено успешно",
    "dashboard_deleted": "Дашборд '{name}' удалён",
    "version_deleted": "Версия {version} дашборда '{name}' удалена",
    "cannot_delete_last_version": "Нельзя удалить единственную версию дашборда",
    "management": "Управление",
    "delete_dashboard_btn": "Удалить весь дашборд",
    "delete_version_btn": "Удалить эту версию"
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

def get_next_version(dashboard_id):
    versions = get_versions(dashboard_id)
    if not versions:
        return 1
    return max(v["version"] for v in versions) + 1

def delete_dashboard(dashboard_id):
    metadata = load_metadata()
    dashboard_versions = metadata[metadata["dashboard_id"] == dashboard_id]
    
    for _, row in dashboard_versions.iterrows():
        file_path = row["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
    
    metadata = metadata[metadata["dashboard_id"] != dashboard_id]
    save_metadata(metadata)
    
    return True

def delete_version(dashboard_id, version_number):
    metadata = load_metadata()
    versions = get_versions(dashboard_id)
    
    if len(versions) <= 1:
        return False, "cannot_delete_last_version"
    
    version_to_delete = metadata[(metadata["dashboard_id"] == dashboard_id) & (metadata["version"] == version_number)]
    
    if not version_to_delete.empty:
        file_path = version_to_delete.iloc[0]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
    
    metadata = metadata[~((metadata["dashboard_id"] == dashboard_id) & (metadata["version"] == version_number))]
    
    metadata.loc[metadata["dashboard_id"] == dashboard_id, "version"] = range(1, len(metadata[metadata["dashboard_id"] == dashboard_id]) + 1)
    
    save_metadata(metadata)
    
    return True, "success"

st.set_page_config(page_title=RUSSIAN_TEXTS["app_title"], layout="wide")
st.title(RUSSIAN_TEXTS["app_title"])
st.markdown("---")

if "show_success" not in st.session_state:
    st.session_state.show_success = False
if "success_message" not in st.session_state:
    st.session_state.success_message = ""
if "show_confirm_delete_dashboard" not in st.session_state:
    st.session_state.show_confirm_delete_dashboard = False
if "show_confirm_delete_version" not in st.session_state:
    st.session_state.show_confirm_delete_version = False
if "dashboard_to_delete" not in st.session_state:
    st.session_state.dashboard_to_delete = None
if "version_to_delete" not in st.session_state:
    st.session_state.version_to_delete = None
if "delete_version_number" not in st.session_state:
    st.session_state.delete_version_number = None

with st.sidebar:
    st.header(RUSSIAN_TEXTS["upload_header"])
    
    dashboard_list = get_dashboard_list()
    
    upload_option = st.radio(
        "Способ загрузки",
        ["К существующему дашборду", "Создать новый дашборд"],
        horizontal=True
    )
    
    if upload_option == "К существующему дашборду":
        if dashboard_list:
            selected_dashboard = st.selectbox(
                RUSSIAN_TEXTS["select_dashboard_to_update"],
                dashboard_list
            )
            dashboard_name_input = selected_dashboard
            next_version = get_next_version(selected_dashboard)
            st.info(f"Будет создана версия {next_version}")
        else:
            st.warning("Нет существующих дашбордов. Создайте новый.")
            dashboard_name_input = None
    else:
        dashboard_name_input = st.text_input(RUSSIAN_TEXTS["new_dashboard_name"])
        if dashboard_name_input:
            next_version = 1
    
    uploaded_file = st.file_uploader(RUSSIAN_TEXTS["screenshot"], type=["png", "jpg", "jpeg"])
    version_description_input = st.text_area(
        RUSSIAN_TEXTS["version_description"],
        placeholder=RUSSIAN_TEXTS["version_description_placeholder"]
    )
    
    if st.button(RUSSIAN_TEXTS["upload_button"]) and dashboard_name_input and uploaded_file:
        timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if upload_option == "К существующему дашборду":
            new_version_number = get_next_version(dashboard_name_input)
        else:
            new_version_number = 1
        
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

        existing_versions = get_versions(dashboard_name_input)
        
        comparison_result = None
        if len(existing_versions) > 1:
            previous_version = existing_versions[-2]
            comparator = DashboardComparator()
            comparison_result = comparator.compare(previous_version["file_path"], filepath)
        
        st.session_state.show_success = True
        st.session_state.success_message = RUSSIAN_TEXTS["upload_success_message"].format(
            name=dashboard_name_input,
            version=new_version_number
        )
        st.session_state.new_version_number = new_version_number
        st.session_state.new_dashboard_name = dashboard_name_input
        st.session_state.comparison_result = comparison_result
        
        st.rerun()

if st.session_state.show_success:
    st.markdown("---")
    st.success(st.session_state.success_message)
    
    if st.session_state.comparison_result:
        result = st.session_state.comparison_result
        st.markdown(f"**{RUSSIAN_TEXTS['changes_from_previous']}:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(RUSSIAN_TEXTS["similarity"], f"{result['similarity'] * 100:.1f}%")
        with col2:
            st.metric(RUSSIAN_TEXTS["changed"], f"{result['change_percentage']:.1f}%")
        with col3:
            st.metric(RUSSIAN_TEXTS["verdict"], result["verdict"])
        st.markdown(f"**{RUSSIAN_TEXTS['analysis']}:** {result['description']}")
    else:
        st.info(RUSSIAN_TEXTS["upload_first"])
    
    if st.button(RUSSIAN_TEXTS["close"]):
        st.session_state.show_success = False
        st.session_state.success_message = ""
        st.session_state.new_version_number = None
        st.session_state.new_dashboard_name = None
        st.session_state.comparison_result = None
        st.rerun()

if st.session_state.show_confirm_delete_dashboard:
    st.markdown("---")
    st.warning(RUSSIAN_TEXTS["confirm_delete_dashboard"].format(name=st.session_state.dashboard_to_delete))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(RUSSIAN_TEXTS["yes_delete"]):
            delete_dashboard(st.session_state.dashboard_to_delete)
            st.session_state.show_confirm_delete_dashboard = False
            st.session_state.dashboard_to_delete = None
            st.success(RUSSIAN_TEXTS["dashboard_deleted"].format(name=st.session_state.dashboard_to_delete))
            st.rerun()
    with col2:
        if st.button(RUSSIAN_TEXTS["cancel"]):
            st.session_state.show_confirm_delete_dashboard = False
            st.session_state.dashboard_to_delete = None
            st.rerun()

if st.session_state.show_confirm_delete_version:
    st.markdown("---")
    st.warning(RUSSIAN_TEXTS["confirm_delete_version"].format(
        version=st.session_state.delete_version_number,
        name=st.session_state.version_to_delete
    ))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(RUSSIAN_TEXTS["yes_delete"]):
            result, error = delete_version(st.session_state.version_to_delete, st.session_state.delete_version_number)
            if result:
                st.session_state.show_confirm_delete_version = False
                st.success(RUSSIAN_TEXTS["version_deleted"].format(
                    version=st.session_state.delete_version_number,
                    name=st.session_state.version_to_delete
                ))
                st.session_state.version_to_delete = None
                st.session_state.delete_version_number = None
                st.rerun()
            else:
                st.error(RUSSIAN_TEXTS[error])
    with col2:
        if st.button(RUSSIAN_TEXTS["cancel"]):
            st.session_state.show_confirm_delete_version = False
            st.session_state.version_to_delete = None
            st.session_state.delete_version_number = None
            st.rerun()

main_tab, compare_tab = st.tabs([RUSSIAN_TEXTS["history_tab"], RUSSIAN_TEXTS["compare_tab"]])

with main_tab:
    st.header(RUSSIAN_TEXTS["history_tab"])

    available_dashboards = get_dashboard_list()
    if not available_dashboards:
        st.info(RUSSIAN_TEXTS["no_dashboards"])
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_dashboard = st.selectbox(RUSSIAN_TEXTS["select_dashboard"], available_dashboards)
        with col2:
            st.markdown("###")
            if st.button(RUSSIAN_TEXTS["delete_dashboard_btn"], use_container_width=True):
                st.session_state.show_confirm_delete_dashboard = True
                st.session_state.dashboard_to_delete = selected_dashboard
                st.rerun()
        
        dashboard_versions = get_versions(selected_dashboard)

        for version_data in reversed(dashboard_versions):
            expander_label = f"{RUSSIAN_TEXTS['version_label']} {version_data['version']} — {version_data['timestamp']}"
            with st.expander(expander_label):
                left_column, middle_column, right_column = st.columns([1, 1, 0.3])

                with left_column:
                    screenshot_image = Image.open(version_data["file_path"])
                    st.image(screenshot_image, caption=f"{RUSSIAN_TEXTS['version_label']} {version_data['version']}", use_container_width=True)

                with middle_column:
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

                with right_column:
                    st.markdown("###")
                    if st.button(RUSSIAN_TEXTS["delete_version_btn"], key=f"del_v{version_data['version']}"):
                        if len(dashboard_versions) <= 1:
                            st.error(RUSSIAN_TEXTS["cannot_delete_last_version"])
                        else:
                            st.session_state.show_confirm_delete_version = True
                            st.session_state.version_to_delete = selected_dashboard
                            st.session_state.delete_version_number = version_data["version"]
                            st.rerun()

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