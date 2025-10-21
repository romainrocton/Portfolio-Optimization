import streamlit as st
from utils.config import dico_pays

def stock_selection_page(indices, index_to_actions):
# UI: stock selection
    
    st.write("")
    selected_display = st.empty()
    
    st.markdown("#### Step 1: Select Indices")

    index_names = list(indices.keys())
    selected_indices = st.multiselect(
        "Select one or more indices:",
        options=index_names,
        default=[],
        key="selected_indices"
    )

    if not selected_indices:
        st.info("Please select at least one index to see the available stocks.")
        return

    # --- STEP 2: Display companies from selected indices ---
    st.markdown("#### Step 2: Select Stocks")

    selected_display = st.empty()
    #st.write("")
    selected_companies = []

    for index_name in selected_indices:
        actions = index_to_actions.get(index_name, [])
        if actions:
            with st.expander(f"**{index_name}** (*{dico_pays[index_name]}*)"):
                selected = st.multiselect(
                    f"Select stocks from {index_name}",
                    options=actions,
                    key=f"ms_{index_name}"
                )
                selected_companies.extend(selected)
        else:
            st.warning(f"No stocks available for {index_name}.")

    # --- Display selected companies summary ---
    if selected_companies:
        selected_display.markdown(
            #"<h4 style='margin-bottom: 10px;'>Selected Stocks:</h4>" +
            "".join([
                f"<span style='font-size:px; background-color:#3a3f51; "
                f"color:#f8f9fa; border-radius:10px; padding:5px 10px; "
                f"margin:3px; display:inline-block;'>{item}</span>"
                for item in selected_companies
            ]),
            unsafe_allow_html=True
        )
    else:
        selected_display.markdown("No stocks selected yet.")
    
    return selected_companies