import streamlit as st
import pandas as pd
import math
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Weekly Target Planner", layout="wide")

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        "uploaded_data": None,
        "agent_data": None,
        "targets_data": None,
        "weekly_average": None,
        "weekly_data": None,
        "recommendations": None,
        "show_recommendations": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

st.title("📦 Weekly Target Planner")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "📊 Weekly Planner", "📈 Dashboard"])

# ====================
# TAB 1: DATA UPLOAD
# ====================
def load_excel_data(uploaded_file):
    """Load and process Excel data with error handling"""
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        # Load sheets
        agent_data = pd.read_excel(xls, sheet_name=0)
        targets_data = pd.read_excel(xls, sheet_name=1)
        weekly_avg_data = pd.read_excel(xls, sheet_name=2)
        
        # Clean and validate weekly average data
        weekly_avg_data.columns = weekly_avg_data.columns.str.strip()
        numeric_cols = ["Tonnage", "Revenue", "Yield"]
        for col in numeric_cols:
            if col in weekly_avg_data.columns:
                weekly_avg_data[col] = pd.to_numeric(weekly_avg_data[col], errors='coerce').fillna(0)
        
        return agent_data, targets_data, weekly_avg_data, None
    except Exception as e:
        return None, None, None, str(e)

# ====================
# TAB 1: DATA UPLOAD
# ====================
with tab1:
    st.header("Upload Database.xlsx File")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    
    if uploaded_file:
        agent_data, targets_data, weekly_avg_data, error = load_excel_data(uploaded_file)
        
        if error:
            st.error(f"❌ Error reading file: {error}")
        else:
            # Store in session state
            st.session_state.uploaded_data = uploaded_file
            st.session_state.agent_data = agent_data
            st.session_state.targets_data = targets_data
            st.session_state.weekly_average = weekly_avg_data

            st.success("✅ File uploaded and data loaded successfully!")
            
            # Display data previews
            with st.expander("📄 Sheet 1: Agent Data", expanded=False):
                st.dataframe(agent_data.head())
            with st.expander("📄 Sheet 2: Targets", expanded=False):
                st.dataframe(targets_data.head())
            with st.expander("📄 Sheet 3: Weekly Average", expanded=False):
                st.dataframe(weekly_avg_data.head())
    else:
        st.info("⬆️ Please upload the latest Database.xlsx file to proceed.")

def get_currency_config(currency):
    """Get currency configuration including rates and symbols"""
    config = {
        "AED": {"rate": 1.0, "symbol": "AED"},
        "USD": {"rate": 1/3.67, "symbol": "$"},
        "BHD": {"rate": 0.102, "symbol": "BHD"}
    }
    return config.get(currency, config["USD"])

def validate_data_availability():
    """Check if required data is loaded"""
    required_data = ["agent_data", "targets_data", "weekly_average"]
    missing = [key for key in required_data if st.session_state.get(key) is None]
    
    if missing:
        st.warning("📁 Please upload the Database.xlsx file in the Data Upload tab first.")
        st.stop()
    return True

def create_metric_box(value, label, background_color="#eeeeee", text_color="black"):
    """Create a styled metric display box"""
    return f"""
    <div style='background-color:{background_color}; padding:20px; border-radius:10px; text-align:center;'>
        <div style='color:{text_color}; font-weight:bold;'>{label}</div>
        <div style='font-size:24px; font-weight:bold;'>{value}</div>
    </div>
    """

def clean_and_validate_data(df):
    """Clean and validate data ensuring consistency and removing invalid values"""
    df_clean = df.copy()
    
    for idx, row in df_clean.iterrows():
        tonnage = float(row['Tonnage']) if pd.notna(row['Tonnage']) else 0
        yield_val = float(row['Yield']) if pd.notna(row['Yield']) else 0
        revenue = float(row['Revenue']) if pd.notna(row['Revenue']) else 0
        
        # Remove negative values - no negative business metrics allowed
        tonnage = max(0, tonnage)
        yield_val = max(0, yield_val)
        revenue = max(0, revenue)
        
        # Apply "all or nothing" rule: if any key metric is 0, all become 0
        if tonnage == 0 or yield_val == 0:
            df_clean.loc[idx, 'Tonnage'] = 0
            df_clean.loc[idx, 'Yield'] = 0
            df_clean.loc[idx, 'Revenue'] = 0
        else:
            # Valid data: ensure revenue = tonnage × yield
            df_clean.loc[idx, 'Tonnage'] = round(tonnage, 0)
            df_clean.loc[idx, 'Yield'] = round(yield_val, 2)
            df_clean.loc[idx, 'Revenue'] = round(tonnage * yield_val, 2)
    
    return df_clean

def calculate_smart_recommendations(week_df, target_tonnage, target_revenue, target_avg_yield):
    """Generate smart recommendations using proportional scaling with target precision"""
    recommendations_df = week_df.copy()
    
    # Get current valid data (exclude zero rows)
    valid_data = recommendations_df[
        (recommendations_df['Tonnage'] > 0) & 
        (recommendations_df['Yield'] > 0) & 
        (recommendations_df['Revenue'] > 0)
    ].copy()
    
    if valid_data.empty:
        # No valid data - create minimal recommendations
        recommendations_df['Tonnage'] = target_tonnage / len(recommendations_df)
        recommendations_df['Yield'] = target_avg_yield
        recommendations_df['Revenue'] = target_revenue / len(recommendations_df)
        return clean_and_validate_data(recommendations_df)
    
    # Current totals
    current_tonnage = valid_data['Tonnage'].sum()
    current_revenue = valid_data['Revenue'].sum()
    current_avg_yield = valid_data['Yield'].mean()
    
    # Strategy 1: Scale tonnage proportionally
    if current_tonnage > 0:
        tonnage_factor = target_tonnage / current_tonnage
        recommendations_df['Tonnage'] = recommendations_df['Tonnage'] * tonnage_factor
    
    # Strategy 2: Adjust yields to meet target average
    if current_avg_yield > 0:
        yield_factor = target_avg_yield / current_avg_yield
        recommendations_df['Yield'] = recommendations_df['Yield'] * yield_factor
    else:
        recommendations_df['Yield'] = target_avg_yield
    
    # Strategy 3: Calculate initial revenue from tonnage × yield
    recommendations_df['Revenue'] = recommendations_df['Tonnage'] * recommendations_df['Yield']
    
    # Strategy 4: Fine-tune revenue to match target exactly
    calculated_revenue = recommendations_df['Revenue'].sum()
    if calculated_revenue > 0:
        revenue_factor = target_revenue / calculated_revenue
        recommendations_df['Revenue'] = recommendations_df['Revenue'] * revenue_factor
        
        # Recalculate yield to maintain consistency
        recommendations_df['Yield'] = recommendations_df.apply(
            lambda row: row['Revenue'] / row['Tonnage'] if row['Tonnage'] > 0 else 0,
            axis=1
        )
    
    # Final validation and cleanup
    return clean_and_validate_data(recommendations_df)

# =====================
# TAB 2: WEEKLY PLANNER
# =====================
with tab2:
    st.header("📊 Weekly Target Planner")
    
    # Validate data availability
    validate_data_availability()

    # Currency & Week selection
    currency = st.selectbox("Select Currency", ["AED", "USD", "BHD"], index=1)
    currency_config = get_currency_config(currency)
    currency_symbol = currency_config["symbol"]
    rate = currency_config["rate"]

    weeks = st.session_state.targets_data["Week"].dropna().unique()
    if len(weeks) == 0:
        st.error("No weeks found in targets data.")
        st.stop()
    
    # Convert weeks to integers to remove decimals for display
    weeks_int = [int(week) for week in weeks if pd.notna(week)]
    weeks_int.sort()  # Sort weeks in ascending order
    
    week_selected = st.selectbox("Select Week", weeks_int)

    # Get targets for week
    tgt_df = st.session_state.targets_data[st.session_state.targets_data["Week"] == week_selected]
    if tgt_df.empty:
        st.warning("⚠️ No target data found for the selected week.")
        st.stop()
    
    tgt = tgt_df.iloc[0]
    orig_ton = tgt["Tgt Wt"]
    orig_yld = tgt["Trgt Yield"]
    orig_rev = tgt["Tgt Rev"]
    
    # Convert targets to selected currency (assuming source is AED)
    aed_rate = 1.0
    conv_tgt_yld = orig_yld / aed_rate * rate
    conv_tgt_rev = orig_rev / aed_rate * rate

    # Initialize or reset weekly_data
    current_weekly_data = st.session_state.weekly_data or {}
    if current_weekly_data.get("week") != week_selected:
        st.session_state.weekly_data = {
            "week": week_selected,
            "current_tonnage": 0.0,
            "current_yield": 0.0,
            "current_revenue": 0.0
        }
    data = st.session_state.weekly_data

    # --- Weekly Targets ---
    st.markdown("### 🎯 Weekly Targets")
    target_cols = st.columns(3)
    target_values = [
        f"{orig_ton:,.0f} kg",
        f"{currency_symbol} {conv_tgt_yld:.2f} / kg",
        f"{currency_symbol} {conv_tgt_rev:,.0f}"
    ]
    target_labels = ["Tonnage", "Yield", "Revenue"]
    
    for col, label, value in zip(target_cols, target_labels, target_values):
        col.markdown(create_metric_box(value, label, "#bbdefb", "#0d47a1"), unsafe_allow_html=True)

    # --- Gap to Target ---
    st.markdown("### 📉 Gap to Target")
    gap_cols = st.columns(3)
    
    gaps = [
        data['current_tonnage'] - orig_ton,
        data['current_yield'] - conv_tgt_yld,
        data['current_revenue'] - conv_tgt_rev
    ]
    gap_labels = ["Tonnage Gap", "Yield Gap", "Revenue Gap"]
    
    for col, label, gap_value in zip(gap_cols, gap_labels, gaps):
        # Format display value
        if "Tonnage" in label:
            display_value = f"{gap_value:,.0f} kg"
        else:
            display_value = f"{currency_symbol} {gap_value:,.2f}"
        
        # Choose background color based on gap (green if positive, red if negative)
        bg_color = "#c8e6c9" if gap_value >= 0 else "#ffcdd2"
        text_color = "#2e7d32" if gap_value >= 0 else "#c62828"
        
        col.markdown(create_metric_box(display_value, label, bg_color, text_color), unsafe_allow_html=True)

    # --- Current Performance ---
    st.markdown("### 📦 Current Performance")
    perf_cols = st.columns(3)
    
    performance_values = [
        f"{data['current_tonnage']:,.0f} kg",
        f"{currency_symbol} {data['current_yield']:.2f} / kg",
        f"{currency_symbol} {data['current_revenue']:,.0f}"
    ]
    performance_labels = ["Tonnage", "Yield", "Revenue"]
    
    for col, label, value in zip(perf_cols, performance_labels, performance_values):
        col.markdown(create_metric_box(value, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Action Buttons ---
    _, btn_col1, btn_col2, btn_col3, btn_col4, btn_col5, _ = st.columns([0.5, 1.2, 1.2, 1.2, 1.2, 1.2, 0.5])
    
    with btn_col1:
        if st.button("Recommend", key="recommend", help="Generate smart recommendations to achieve weekly targets with data integrity"):
            wa = st.session_state.weekly_average.copy()
            week_df = wa[wa['Week'] == week_selected].copy()
            
            if not week_df.empty:
                try:
                    # Define targets
                    target_tonnage = orig_ton
                    target_revenue = conv_tgt_rev
                    target_avg_yield = conv_tgt_yld
                    
                    # Generate smart recommendations using the new function
                    recommendations_df = calculate_smart_recommendations(
                        week_df, target_tonnage, target_revenue, target_avg_yield
                    )
                    
                    # Store recommendations
                    st.session_state.recommendations = recommendations_df
                    st.session_state.show_recommendations = True
                    
                    # Calculate and display results
                    final_tonnage = recommendations_df['Tonnage'].sum()
                    final_revenue = recommendations_df['Revenue'].sum()
                    final_yields = recommendations_df[recommendations_df['Yield'] > 0]['Yield']
                    final_avg_yield = final_yields.mean() if len(final_yields) > 0 else 0
                    
                    # Success message with detailed results
                    st.success("✅ Smart recommendations generated with data integrity!")
                    st.info(f"📊 **Results:** Tonnage: {final_tonnage:,.0f}/{target_tonnage:,.0f} | Revenue: {currency_symbol}{final_revenue:,.0f}/{currency_symbol}{target_revenue:,.0f} | Avg Yield: {final_avg_yield:.2f}/{target_avg_yield:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
            else:
                st.error("❌ No weekly average data found for the selected week.")
    
    with btn_col2:
        if st.button("Adjust", key="adjust", help="Clean and adjust data ensuring mathematical consistency and removing invalid values"):
            try:
                # Check if we're working with recommendations or weekly average
                if st.session_state.get("show_recommendations", False) and "recommendations" in st.session_state:
                    # Clean and adjust recommendations table
                    recommendations_df = st.session_state.recommendations.copy()
                    cleaned_df = clean_and_validate_data(recommendations_df)
                    st.session_state.recommendations = cleaned_df
                    st.success("✅ Recommendations cleaned: data validated, negatives removed, consistency ensured.")
                else:
                    # Clean and adjust weekly average table
                    wa = st.session_state.weekly_average.copy()
                    week_df = wa[wa['Week'] == week_selected].copy()
                    other_weeks = wa[wa['Week'] != week_selected]
                    
                    cleaned_week_df = clean_and_validate_data(week_df)
                    st.session_state.weekly_average = pd.concat([other_weeks, cleaned_week_df], ignore_index=True)
                    st.success("✅ Weekly average cleaned: data validated, negatives removed, consistency ensured.")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during data adjustment: {str(e)}")
    
    with btn_col3:
        if st.button("Apply", key="apply", help="Apply the current table values (recommendations or weekly average) to update current performance metrics"):
            # Check if we're applying recommendations or weekly average
            if st.session_state.get("show_recommendations", False) and "recommendations" in st.session_state:
                # Apply recommendations
                recommendations_df = st.session_state.recommendations
                
                total_tonnage = recommendations_df['Tonnage'].sum()
                total_revenue = recommendations_df['Revenue'].sum()
                # Calculate average yield from all agents in the table
                valid_yields = recommendations_df[recommendations_df['Yield'] > 0]['Yield']
                avg_yield = valid_yields.mean() if len(valid_yields) > 0 else 0
                
                # Update current performance
                data['current_tonnage'] = total_tonnage
                data['current_revenue'] = total_revenue
                data['current_yield'] = avg_yield
                
                st.success("✅ Recommendations applied to current performance.")
            else:
                # Apply weekly average
                wa = st.session_state.weekly_average
                week_df = wa[wa['Week'] == week_selected]
                
                total_tonnage = week_df['Tonnage'].sum()
                total_revenue = week_df['Revenue'].sum()
                # Calculate average yield from all agents in the table
                valid_yields = week_df[week_df['Yield'] > 0]['Yield']
                avg_yield = valid_yields.mean() if len(valid_yields) > 0 else 0
                
                # Update current performance
                data['current_tonnage'] = total_tonnage
                data['current_revenue'] = total_revenue
                data['current_yield'] = avg_yield
                
                st.success("✅ Weekly average applied to current performance.")
            
            st.rerun()  # Refresh to show updated values

    with btn_col4:
        if st.button("Reset", key="reset", help="Reset all current performance values (tonnage, yield, revenue) back to zero"):
            # Reset current performance to zero
            data['current_tonnage'] = 0.0
            data['current_revenue'] = 0.0
            data['current_yield'] = 0.0
            
            st.success("✅ Current performance values reset to zero.")
            st.rerun()  # Refresh to show updated values

    with btn_col5:
        # Export functionality
        if st.button("Export", key="export", help="Export the current Weekly Average table to Excel file"):
            try:
                # Determine what to export based on current view
                if st.session_state.get("show_recommendations", False) and "recommendations" in st.session_state:
                    # Export recommendations table
                    export_data = st.session_state.recommendations.copy()
                    table_type = "Recommendations"
                    filename_prefix = "recommendations"
                else:
                    # Export weekly average table for current week
                    wa = st.session_state.weekly_average
                    export_data = wa[wa['Week'] == week_selected].copy()
                    table_type = "Weekly Average"
                    filename_prefix = "weekly_average"
                
                if not export_data.empty:
                    # Create Excel file in memory
                    output = io.BytesIO()
                    
                    # Export to Excel
                    export_data.to_excel(output, index=False, engine='openpyxl')
                    output.seek(0)
                    
                    # Generate filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{filename_prefix}_{week_selected}_{timestamp}.xlsx"
                    
                    st.download_button(
                        label=f"📥 Download {table_type} Excel",
                        data=output.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help=f"Click to download the {table_type.lower()} table as Excel file"
                    )
                    
                    st.success(f"✅ {table_type} table ready for download! File: {filename}")
                else:
                    st.warning("⚠️ No data available to export for the selected week.")
                    
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")

    # --- Editable Weekly Average Table ---
    st.markdown("### 📈 Weekly Average")
    
    # Show recommendations table if available, otherwise show weekly average
    if st.session_state.get("show_recommendations", False) and "recommendations" in st.session_state:
        st.info("📊 Showing recommendations to meet targets. Click 'Apply' to use these values.")
        
        # Display editable recommendations table
        edited_recommendations = st.data_editor(
            st.session_state.recommendations, 
            key='recommendations_editor', 
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Tonnage": st.column_config.NumberColumn(
                    "Tonnage",
                    help="Recommended tonnage in kg",
                    format="%.0f"
                ),
                "Revenue": st.column_config.NumberColumn(
                    "Revenue", 
                    help="Recommended revenue amount",
                    format="%.0f"
                ),
                "Yield": st.column_config.NumberColumn(
                    "Yield",
                    help="Recommended yield per kg", 
                    format="%.2f"
                )
            }
        )
        
        # Update recommendations in session state with edited data
        st.session_state.recommendations = edited_recommendations
        
        # Add back to weekly average button
        col1, _, _ = st.columns([2, 2, 6])
        with col1:
            if st.button("📋 Back to Weekly Average", key="back_to_weekly"):
                st.session_state.show_recommendations = False
                st.rerun()
            
    elif st.session_state.weekly_average is not None:
        wa = st.session_state.weekly_average.copy()
        week_data = wa[wa['Week'] == week_selected]
        
        if not week_data.empty:
            # Make the table editable
            edited_data = st.data_editor(
                week_data, 
                key='weekly_avg_editor', 
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Tonnage": st.column_config.NumberColumn(
                        "Tonnage",
                        help="Tonnage in kg",
                        format="%.0f"
                    ),
                    "Revenue": st.column_config.NumberColumn(
                        "Revenue", 
                        help="Revenue amount",
                        format="%.0f"
                    ),
                    "Yield": st.column_config.NumberColumn(
                        "Yield",
                        help="Yield per kg", 
                        format="%.2f"
                    )
                }
            )
            
            # Update session state with edited data
            other_weeks = wa[wa['Week'] != week_selected]
            st.session_state.weekly_average = pd.concat([other_weeks, edited_data], ignore_index=True)
        else:
            st.info("No weekly average data found for the selected week.")
    else:
        st.warning("No weekly average data available. Please upload data first.")

# ====================
# TAB 3: DASHBOARD
# ====================
with tab3:
    st.header("📈 Data Analytics Dashboard")
    
    # Validate data availability
    validate_data_availability()
    
    # Dashboard currency selection
    dashboard_currency = st.selectbox("Dashboard Currency", ["AED", "USD", "BHD"], index=1, key="dashboard_currency")
    dashboard_config = get_currency_config(dashboard_currency)
    dashboard_symbol = dashboard_config["symbol"]
    dashboard_rate = dashboard_config["rate"]
    
    # =============================
    # SECTION 1: KEY METRICS
    # =============================
    st.markdown("### 📊 Key Performance Indicators")
    
    # Calculate overall statistics
    try:
        # Agent statistics
        total_agents = len(st.session_state.agent_data) if st.session_state.agent_data is not None else 0
        
        # Weekly average statistics
        wa = st.session_state.weekly_average
        total_weeks = len(wa['Week'].unique()) if wa is not None and 'Week' in wa.columns else 0
        
        # Targets statistics
        targets = st.session_state.targets_data
        total_target_tonnage = targets['Tgt Wt'].sum() if targets is not None and 'Tgt Wt' in targets.columns else 0
        total_target_revenue = (targets['Tgt Rev'].sum() * dashboard_rate) if targets is not None and 'Tgt Rev' in targets.columns else 0
        avg_target_yield = (targets['Trgt Yield'].mean() * dashboard_rate) if targets is not None and 'Trgt Yield' in targets.columns else 0
        
        # Weekly average totals
        if wa is not None and len(wa) > 0:
            total_wa_tonnage = wa['Tonnage'].sum()
            total_wa_revenue = wa['Revenue'].sum() * dashboard_rate
            avg_wa_yield = wa[wa['Yield'] > 0]['Yield'].mean() * dashboard_rate if len(wa[wa['Yield'] > 0]) > 0 else 0
            
            # Calculate unique destinations if column exists
            total_destinations = len(wa['Destination'].unique()) if 'Destination' in wa.columns else 0
        else:
            total_wa_tonnage = total_wa_revenue = avg_wa_yield = total_destinations = 0
        
        # Display KPI metrics
        kpi_cols = st.columns(4)
        
        kpi_data = [
            (f"{total_agents:,}", "Total Agents", "#e3f2fd"),
            (f"{total_weeks:,}", "Total Weeks", "#f3e5f5"),
            (f"{total_destinations:,}", "Destinations", "#e8f5e8"),
            (f"{dashboard_symbol}{total_target_revenue:,.0f}", "Total Targets", "#fff3e0")
        ]
        
        for col, (value, label, color) in zip(kpi_cols, kpi_data):
            col.markdown(create_metric_box(value, label, color, "#1976d2"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance comparison metrics
        perf_cols = st.columns(3)
        
        perf_data = [
            (f"{total_wa_tonnage:,.0f} kg", f"Current Tonnage\n(Target: {total_target_tonnage:,.0f} kg)"),
            (f"{dashboard_symbol}{total_wa_revenue:,.0f}", f"Current Revenue\n(Target: {dashboard_symbol}{total_target_revenue:,.0f})"),
            (f"{dashboard_symbol}{avg_wa_yield:.2f}/kg", f"Avg Yield\n(Target: {dashboard_symbol}{avg_target_yield:.2f}/kg)")
        ]
        
        for col, (value, label) in zip(perf_cols, perf_data):
            col.markdown(create_metric_box(value, label, "#f5f5f5", "#424242"), unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error calculating KPIs: {str(e)}")
    
    st.markdown("---")
    
    # =============================
    # SECTION 2: CHARTS & ANALYSIS
    # =============================
    st.markdown("### 📊 Visual Analytics")
    
    # Create visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["📈 Weekly Trends", "🥧 Distribution Analysis", "📊 Performance Comparison", "🎯 Achievement Tracking"])
    
    try:
        # ===== TAB 1: WEEKLY TRENDS =====
        with viz_tab1:
            st.markdown("#### 📈 Weekly Performance Trends")
            
            if targets is not None and wa is not None:
                # Prepare trend data
                trend_data = []
                
                # Get available weeks from both datasets
                target_weeks = set(targets['Week'].dropna().unique())
                wa_weeks = set(wa['Week'].dropna().unique()) if 'Week' in wa.columns else set()
                common_weeks = target_weeks.intersection(wa_weeks) if wa_weeks else target_weeks
                
                for week in sorted(common_weeks):
                    try:
                        # Get week targets
                        week_targets_df = targets[targets['Week'] == week]
                        if len(week_targets_df) == 0:
                            continue
                        week_targets = week_targets_df.iloc[0]
                        
                        # Get week weekly average data
                        week_wa = wa[wa['Week'] == week] if wa is not None else pd.DataFrame()
                        
                        current_tonnage = week_wa['Tonnage'].sum() if len(week_wa) > 0 and 'Tonnage' in week_wa.columns else 0
                        current_revenue = week_wa['Revenue'].sum() * dashboard_rate if len(week_wa) > 0 and 'Revenue' in week_wa.columns else 0
                        current_yield = week_wa[week_wa['Yield'] > 0]['Yield'].mean() * dashboard_rate if len(week_wa) > 0 and 'Yield' in week_wa.columns and len(week_wa[week_wa['Yield'] > 0]) > 0 else 0
                        
                        trend_data.append({
                            'Week': int(week),
                            'Target Tonnage': week_targets.get('Tgt Wt', 0),
                            'Current Tonnage': current_tonnage,
                            'Target Revenue': week_targets.get('Tgt Rev', 0) * dashboard_rate,
                            'Current Revenue': current_revenue,
                            'Target Yield': week_targets.get('Trgt Yield', 0) * dashboard_rate,
                            'Current Yield': current_yield
                        })
                    except Exception as week_error:
                        st.warning(f"Error processing week {week}: {str(week_error)}")
                        continue
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    
                    # Line chart for tonnage trends
                    fig_tonnage = go.Figure()
                    fig_tonnage.add_trace(go.Scatter(
                        x=trend_df['Week'], 
                        y=trend_df['Target Tonnage'],
                        mode='lines+markers',
                        name='Target Tonnage',
                        line=dict(color='#ff7f0e', width=3)
                    ))
                    fig_tonnage.add_trace(go.Scatter(
                        x=trend_df['Week'], 
                        y=trend_df['Current Tonnage'],
                        mode='lines+markers',
                        name='Current Tonnage',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    fig_tonnage.update_layout(
                        title="Weekly Tonnage Trends",
                        xaxis_title="Week",
                        yaxis_title="Tonnage (kg)",
                        height=400
                    )
                    st.plotly_chart(fig_tonnage, use_container_width=True)
                    
                    # Line chart for revenue trends
                    fig_revenue = go.Figure()
                    fig_revenue.add_trace(go.Scatter(
                        x=trend_df['Week'], 
                        y=trend_df['Target Revenue'],
                        mode='lines+markers',
                        name='Target Revenue',
                        line=dict(color='#2ca02c', width=3)
                    ))
                    fig_revenue.add_trace(go.Scatter(
                        x=trend_df['Week'], 
                        y=trend_df['Current Revenue'],
                        mode='lines+markers',
                        name='Current Revenue',
                        line=dict(color='#d62728', width=3)
                    ))
                    fig_revenue.update_layout(
                        title=f"Weekly Revenue Trends ({dashboard_currency})",
                        xaxis_title="Week",
                        yaxis_title=f"Revenue ({dashboard_symbol})",
                        height=400
                    )
                    st.plotly_chart(fig_revenue, use_container_width=True)
                else:
                    st.info("No matching week data found for trend analysis")
            else:
                st.info("No data available for trend analysis")
        
        # ===== TAB 2: DISTRIBUTION ANALYSIS =====
        with viz_tab2:
            st.markdown("#### 🥧 Data Distribution Analysis")
            
            # Debug: Show available columns
            if wa is not None and len(wa) > 0:
                st.info(f"📋 Available columns in weekly average data: {list(wa.columns)}")
                
                # Find the best grouping column
                grouping_column = None
                if 'Agent' in wa.columns:
                    grouping_column = 'Agent'
                elif 'Destination' in wa.columns:
                    grouping_column = 'Destination'
                elif 'agent' in wa.columns:
                    grouping_column = 'agent'
                elif 'destination' in wa.columns:
                    grouping_column = 'destination'
                else:
                    # Look for any column that could be used for grouping (non-numeric columns)
                    non_numeric_cols = []
                    for col in wa.columns:
                        if col not in ['Week', 'Tonnage', 'Revenue', 'Yield'] and wa[col].dtype == 'object':
                            non_numeric_cols.append(col)
                    
                    if non_numeric_cols:
                        grouping_column = non_numeric_cols[0]  # Use the first non-numeric column
                        st.info(f"🔍 Using '{grouping_column}' as grouping column")
                
                if grouping_column:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Tonnage distribution pie chart
                        if 'Tonnage' in wa.columns:
                            group_tonnage = wa.groupby(grouping_column)['Tonnage'].sum().reset_index()
                            group_tonnage = group_tonnage[group_tonnage['Tonnage'] > 0].sort_values('Tonnage', ascending=False).head(10)
                            
                            if len(group_tonnage) > 0:
                                fig_pie_tonnage = px.pie(
                                    group_tonnage, 
                                    values='Tonnage', 
                                    names=grouping_column,
                                    title=f'Tonnage Distribution by {grouping_column}'
                                )
                                fig_pie_tonnage.update_layout(height=500)
                                st.plotly_chart(fig_pie_tonnage, use_container_width=True)
                            else:
                                st.info(f"No {grouping_column.lower()} tonnage data available")
                        else:
                            st.info("No Tonnage column found")
                    
                    with col2:
                        # Revenue distribution pie chart
                        if 'Revenue' in wa.columns:
                            group_revenue = wa.groupby(grouping_column)['Revenue'].sum().reset_index()
                            group_revenue['Revenue'] *= dashboard_rate
                            group_revenue = group_revenue[group_revenue['Revenue'] > 0].sort_values('Revenue', ascending=False).head(10)
                            
                            if len(group_revenue) > 0:
                                fig_pie_revenue = px.pie(
                                    group_revenue, 
                                    values='Revenue', 
                                    names=grouping_column,
                                    title=f'Revenue Distribution by {grouping_column} ({dashboard_currency})'
                                )
                                fig_pie_revenue.update_layout(height=500)
                                st.plotly_chart(fig_pie_revenue, use_container_width=True)
                            else:
                                st.info(f"No {grouping_column.lower()} revenue data available")
                        else:
                            st.info("No Revenue column found")
                else:
                    st.warning("⚠️ No suitable grouping column found for distribution analysis. Available columns: " + ", ".join(wa.columns))
                    st.info("💡 Expected columns: Agent, Destination, or any text column for grouping")
            else:
                st.info("No weekly average data available")
        
        # ===== TAB 3: PERFORMANCE COMPARISON =====
        with viz_tab3:
            st.markdown("#### 📊 Performance Comparison Charts")
            
            if wa is not None and len(wa) > 0:
                # Find the best grouping column
                grouping_column = None
                if 'Agent' in wa.columns:
                    grouping_column = 'Agent'
                elif 'Destination' in wa.columns:
                    grouping_column = 'Destination'
                elif 'agent' in wa.columns:
                    grouping_column = 'agent'
                elif 'destination' in wa.columns:
                    grouping_column = 'destination'
                else:
                    # Look for any column that could be used for grouping (non-numeric columns)
                    non_numeric_cols = []
                    for col in wa.columns:
                        if col not in ['Week', 'Tonnage', 'Revenue', 'Yield'] and wa[col].dtype == 'object':
                            non_numeric_cols.append(col)
                    
                    if non_numeric_cols:
                        grouping_column = non_numeric_cols[0]  # Use the first non-numeric column
                
                if grouping_column:
                    # Performance analysis by grouping column
                    performance_df = wa.groupby(grouping_column).agg({
                        'Tonnage': 'sum',
                        'Revenue': 'sum',
                        'Yield': 'mean'
                    }).round(2)
                    performance_df['Revenue'] *= dashboard_rate
                    performance_df['Yield'] *= dashboard_rate
                    performance_df = performance_df[performance_df['Revenue'] > 0].sort_values('Revenue', ascending=False).head(15)
                    performance_df = performance_df.reset_index()
                    
                    if len(performance_df) > 0:
                        if grouping_column.lower() in ['agent', 'agents']:
                            # Multi-metric bar chart for agents
                            fig_bar = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=(f'Revenue by {grouping_column}', f'Tonnage by {grouping_column}', f'Yield by {grouping_column}', 'Performance Overview'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                      [{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            # Revenue bar chart
                            fig_bar.add_trace(
                                go.Bar(x=performance_df[grouping_column], y=performance_df['Revenue'], 
                                      name='Revenue', marker_color='#2ca02c'),
                                row=1, col=1
                            )
                            
                            # Tonnage bar chart
                            fig_bar.add_trace(
                                go.Bar(x=performance_df[grouping_column], y=performance_df['Tonnage'], 
                                      name='Tonnage', marker_color='#1f77b4'),
                                row=1, col=2
                            )
                            
                            # Yield bar chart
                            fig_bar.add_trace(
                                go.Bar(x=performance_df[grouping_column], y=performance_df['Yield'], 
                                      name='Yield', marker_color='#ff7f0e'),
                                row=2, col=1
                            )
                            
                            # Combined scatter plot
                            fig_bar.add_trace(
                                go.Scatter(x=performance_df['Tonnage'], y=performance_df['Revenue'], 
                                          mode='markers', name='Performance',
                                          marker=dict(size=performance_df['Yield']*2, color='#d62728')),
                                row=2, col=2
                            )
                            
                            fig_bar.update_layout(height=800, showlegend=False)
                            fig_bar.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        else:
                            # Horizontal bar chart for destinations or other groupings
                            fig_dest = go.Figure()
                            fig_dest.add_trace(go.Bar(
                                y=performance_df[grouping_column],
                                x=performance_df['Revenue'],
                                orientation='h',
                                name='Revenue',
                                marker_color='#2ca02c'
                            ))
                            fig_dest.update_layout(
                                title=f'Revenue by {grouping_column} ({dashboard_currency})',
                                height=600,
                                xaxis_title=f'Revenue ({dashboard_symbol})',
                                yaxis_title=grouping_column
                            )
                            st.plotly_chart(fig_dest, use_container_width=True)
                            
                            # Additional charts
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Tonnage bar chart
                                fig_tonnage = px.bar(
                                    performance_df.head(10), 
                                    x=grouping_column, 
                                    y='Tonnage',
                                    title=f'Top 10 {grouping_column} by Tonnage',
                                    color='Tonnage',
                                    color_continuous_scale='Blues'
                                )
                                fig_tonnage.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_tonnage, use_container_width=True)
                            
                            with col2:
                                # Yield bar chart
                                fig_yield = px.bar(
                                    performance_df.head(10), 
                                    x=grouping_column, 
                                    y='Yield',
                                    title=f'Top 10 {grouping_column} by Yield',
                                    color='Yield',
                                    color_continuous_scale='Oranges'
                                )
                                fig_yield.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_yield, use_container_width=True)
                    else:
                        st.info(f"No {grouping_column.lower()} performance data available")
                else:
                    st.warning("⚠️ No suitable grouping column found for performance comparison. Available columns: " + ", ".join(wa.columns))
                    st.info("💡 Expected columns: Agent, Destination, or any text column for grouping")
                    
                    # Show a simple summary table instead
                    if 'Tonnage' in wa.columns and 'Revenue' in wa.columns and 'Yield' in wa.columns:
                        st.markdown("#### 📋 Overall Performance Summary")
                        summary_data = {
                            'Metric': ['Total Tonnage', 'Total Revenue', 'Average Yield'],
                            'Value': [
                                f"{wa['Tonnage'].sum():,.0f} kg",
                                f"{dashboard_symbol}{wa['Revenue'].sum() * dashboard_rate:,.0f}",
                                f"{dashboard_symbol}{wa['Yield'].mean() * dashboard_rate:.2f}/kg"
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("No weekly average data available")
        
        # ===== TAB 4: ACHIEVEMENT TRACKING =====
        with viz_tab4:
            st.markdown("#### 🎯 Achievement vs Target Analysis")
            
            if targets is not None and wa is not None:
                # Calculate achievement percentages
                achievement_data = []
                
                # Get available weeks from both datasets
                target_weeks = set(targets['Week'].dropna().unique())
                wa_weeks = set(wa['Week'].dropna().unique()) if 'Week' in wa.columns else set()
                common_weeks = target_weeks.intersection(wa_weeks) if wa_weeks else target_weeks
                
                for week in sorted(common_weeks):
                    try:
                        # Get week targets
                        week_targets_df = targets[targets['Week'] == week]
                        if len(week_targets_df) == 0:
                            continue
                        week_targets = week_targets_df.iloc[0]
                        
                        # Get week weekly average data
                        week_wa = wa[wa['Week'] == week] if wa is not None else pd.DataFrame()
                        
                        if len(week_wa) > 0:
                            current_tonnage = week_wa['Tonnage'].sum() if 'Tonnage' in week_wa.columns else 0
                            current_revenue = week_wa['Revenue'].sum() * dashboard_rate if 'Revenue' in week_wa.columns else 0
                            current_yield = week_wa[week_wa['Yield'] > 0]['Yield'].mean() * dashboard_rate if 'Yield' in week_wa.columns and len(week_wa[week_wa['Yield'] > 0]) > 0 else 0
                            
                            target_tonnage = week_targets.get('Tgt Wt', 0)
                            target_revenue = week_targets.get('Tgt Rev', 0) * dashboard_rate
                            target_yield = week_targets.get('Trgt Yield', 0) * dashboard_rate
                            
                            tonnage_achievement = (current_tonnage / target_tonnage * 100) if target_tonnage > 0 else 0
                            revenue_achievement = (current_revenue / target_revenue * 100) if target_revenue > 0 else 0
                            yield_achievement = (current_yield / target_yield * 100) if target_yield > 0 else 0
                            
                            achievement_data.append({
                                'Week': int(week),
                                'Tonnage Achievement': round(tonnage_achievement, 1),
                                'Revenue Achievement': round(revenue_achievement, 1),
                                'Yield Achievement': round(yield_achievement, 1),
                                'Overall Achievement': round((tonnage_achievement + revenue_achievement + yield_achievement) / 3, 1)
                            })
                    except Exception as week_error:
                        st.warning(f"Error processing achievement for week {week}: {str(week_error)}")
                        continue
                
                if achievement_data:
                    achievement_df = pd.DataFrame(achievement_data)
                    
                    # Achievement bar chart
                    fig_achievement = go.Figure()
                    
                    fig_achievement.add_trace(go.Bar(
                        x=achievement_df['Week'],
                        y=achievement_df['Tonnage Achievement'],
                        name='Tonnage Achievement',
                        marker_color='#1f77b4'
                    ))
                    
                    fig_achievement.add_trace(go.Bar(
                        x=achievement_df['Week'],
                        y=achievement_df['Revenue Achievement'],
                        name='Revenue Achievement',
                        marker_color='#2ca02c'
                    ))
                    
                    fig_achievement.add_trace(go.Bar(
                        x=achievement_df['Week'],
                        y=achievement_df['Yield Achievement'],
                        name='Yield Achievement',
                        marker_color='#ff7f0e'
                    ))
                    
                    # Add target line at 100%
                    fig_achievement.add_hline(y=100, line_dash="dash", line_color="red", 
                                            annotation_text="Target (100%)")
                    
                    fig_achievement.update_layout(
                        title='Achievement Percentage by Week',
                        xaxis_title='Week',
                        yaxis_title='Achievement Percentage (%)',
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig_achievement, use_container_width=True)
                    
                    # Overall achievement radar chart
                    if len(achievement_df) > 0:
                        avg_achievements = {
                            'Tonnage': achievement_df['Tonnage Achievement'].mean(),
                            'Revenue': achievement_df['Revenue Achievement'].mean(),
                            'Yield': achievement_df['Yield Achievement'].mean(),
                            'Overall': achievement_df['Overall Achievement'].mean()
                        }
                        
                        fig_radar = go.Figure()
                        
                        categories = list(avg_achievements.keys())
                        values = list(avg_achievements.values())
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values + [values[0]],  # Close the radar chart
                            theta=categories + [categories[0]],
                            fill='toself',
                            name='Average Achievement'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 150]
                                )),
                            showlegend=True,
                            title="Overall Achievement Radar",
                            height=500
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("No achievement data available for common weeks")
            else:
                st.info("Targets and weekly data needed for achievement analysis")
                
    except Exception as e:
        st.error(f"❌ Error generating visualizations: {str(e)}")
    
    st.markdown("---")
    
    # =============================
    # SECTION 3: DATA TABLES
    # =============================
    st.markdown("### 📋 Detailed Data Tables")
    
    table_tab1, table_tab2, table_tab3 = st.tabs(["📊 Targets Summary", "📈 Weekly Performance", "🎯 Achievement Analysis"])
    
    with table_tab1:
        st.markdown("#### 🎯 All Weekly Targets")
        if targets is not None:
            display_targets = targets.copy()
            # Convert to selected currency
            if 'Tgt Rev' in display_targets.columns:
                display_targets['Tgt Rev'] = display_targets['Tgt Rev'] * dashboard_rate
            if 'Trgt Yield' in display_targets.columns:
                display_targets['Trgt Yield'] = display_targets['Trgt Yield'] * dashboard_rate
            
            st.dataframe(
                display_targets,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Week": st.column_config.NumberColumn("Week", format="%.0f"),
                    "Tgt Wt": st.column_config.NumberColumn("Target Weight (kg)", format="%.0f"),
                    "Trgt Yield": st.column_config.NumberColumn("Target Yield", format=f"{dashboard_symbol}%.2f"),
                    "Tgt Rev": st.column_config.NumberColumn("Target Revenue", format=f"{dashboard_symbol}%.0f")
                }
            )
        else:
            st.info("No targets data available")
    
    with table_tab2:
        st.markdown("#### 📈 Weekly Average Performance")
        if wa is not None:
            display_wa = wa.copy()
            # Convert to selected currency
            if 'Revenue' in display_wa.columns:
                display_wa['Revenue'] = display_wa['Revenue'] * dashboard_rate
            if 'Yield' in display_wa.columns:
                display_wa['Yield'] = display_wa['Yield'] * dashboard_rate
            
            st.dataframe(
                display_wa,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Week": st.column_config.NumberColumn("Week", format="%.0f"),
                    "Tonnage": st.column_config.NumberColumn("Tonnage (kg)", format="%.0f"),
                    "Revenue": st.column_config.NumberColumn("Revenue", format=f"{dashboard_symbol}%.0f"),
                    "Yield": st.column_config.NumberColumn("Yield", format=f"{dashboard_symbol}%.2f")
                }
            )
        else:
            st.info("No weekly average data available")
    
    with table_tab3:
        st.markdown("#### 🎯 Achievement vs Targets")
        try:
            if targets is not None and wa is not None:
                achievement_data = []
                
                # Get available weeks from both datasets
                target_weeks = set(targets['Week'].dropna().unique())
                wa_weeks = set(wa['Week'].dropna().unique()) if 'Week' in wa.columns else set()
                common_weeks = target_weeks.intersection(wa_weeks) if wa_weeks else target_weeks
                
                for week in sorted(common_weeks):
                    try:
                        # Get week targets
                        week_targets_df = targets[targets['Week'] == week]
                        if len(week_targets_df) == 0:
                            continue
                        week_targets = week_targets_df.iloc[0]
                        
                        # Get week weekly average data
                        week_wa = wa[wa['Week'] == week] if wa is not None else pd.DataFrame()
                        
                        if len(week_wa) > 0:
                            current_tonnage = week_wa['Tonnage'].sum() if 'Tonnage' in week_wa.columns else 0
                            current_revenue = week_wa['Revenue'].sum() * dashboard_rate if 'Revenue' in week_wa.columns else 0
                            current_yield = week_wa[week_wa['Yield'] > 0]['Yield'].mean() * dashboard_rate if 'Yield' in week_wa.columns and len(week_wa[week_wa['Yield'] > 0]) > 0 else 0
                            
                            target_tonnage = week_targets.get('Tgt Wt', 0)
                            target_revenue = week_targets.get('Tgt Rev', 0) * dashboard_rate
                            target_yield = week_targets.get('Trgt Yield', 0) * dashboard_rate
                            
                            tonnage_achievement = (current_tonnage / target_tonnage * 100) if target_tonnage > 0 else 0
                            revenue_achievement = (current_revenue / target_revenue * 100) if target_revenue > 0 else 0
                            yield_achievement = (current_yield / target_yield * 100) if target_yield > 0 else 0
                            
                            achievement_data.append({
                                'Week': int(week),
                                'Tonnage Achievement %': round(tonnage_achievement, 1),
                                'Revenue Achievement %': round(revenue_achievement, 1),
                                'Yield Achievement %': round(yield_achievement, 1),
                                'Overall Achievement %': round((tonnage_achievement + revenue_achievement + yield_achievement) / 3, 1)
                            })
                    except Exception as week_error:
                        st.warning(f"Error processing achievement table for week {week}: {str(week_error)}")
                        continue
                
                if achievement_data:
                    achievement_df = pd.DataFrame(achievement_data)
                    st.dataframe(
                        achievement_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Week": st.column_config.NumberColumn("Week", format="%.0f"),
                            "Tonnage Achievement %": st.column_config.NumberColumn("Tonnage %", format="%.1f%%"),
                            "Revenue Achievement %": st.column_config.NumberColumn("Revenue %", format="%.1f%%"),
                            "Yield Achievement %": st.column_config.NumberColumn("Yield %", format="%.1f%%"),
                            "Overall Achievement %": st.column_config.NumberColumn("Overall %", format="%.1f%%")
                        }
                    )
                else:
                    st.info("No achievement data available for common weeks")
            else:
                st.info("Targets and weekly data needed for achievement analysis")
        except Exception as e:
            st.error(f"❌ Error calculating achievements: {str(e)}")

# end tab3
