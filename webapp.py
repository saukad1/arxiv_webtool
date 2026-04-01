import streamlit as st
import pandas as pd
import math

# 1. SETUP & CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="arXiv filter", layout="wide")
BUCKET_NAME = "my-arxiv-parquet-bucket"
PARQUET_PATH = f"gs://{BUCKET_NAME}/df_streamlit.parquet"

# Load Google Credentials from Streamlit's secure secrets
gcp_credentials = dict(st.secrets["gcp_service_account"])

# 2. LOAD DATA
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_parquet(PARQUET_PATH, storage_options={"token": gcp_credentials})
    df['date_only'] = pd.to_datetime(df['date_only']).dt.date
    return df
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df
min_date = df['date_only'].min()
max_date = df['date_only'].max()

def toggle_read(paper_id):
    """Updates the main dataframe and saves to disk when a checkbox is clicked."""
    # Get the new boolean value from the checkbox's session state
    is_read = st.session_state[f"chk_{paper_id}"]
    
    # Update the session state dataframe (convert boolean to 1 or 0)
    st.session_state.df.loc[st.session_state.df['id'] == paper_id, 'read'] = int(is_read)
    
    # Save the updated dataframe back to the parquet file so progress isn't lost
    st.session_state.df.to_parquet(PARQUET_PATH, engine='pyarrow', storage_options={"token": gcp_credentials})
    
    st.toast("Successfully updated!",icon=":material/thumb_up:", duration="short")
    st.cache_data.clear()
    
    # Remember this paper's ID so we can keep its expander open
    st.session_state.last_expanded = paper_id

def toggle_star(paper_id):
    """Updates the main dataframe and saves to disk when a checkbox is clicked."""
    # Get the new boolean value from the checkbox's session state
    is_read = st.session_state[f"star_{paper_id}"]
    
    # Update the session state dataframe (convert boolean to 1 or 0)
    st.session_state.df.loc[st.session_state.df['id'] == paper_id, 'star'] = int(is_read)
    
    # Save the updated dataframe back to the parquet file so progress isn't lost
    st.session_state.df.to_parquet(PARQUET_PATH, engine='pyarrow', storage_options={"token": gcp_credentials})

    st.toast("Successfully updated!",icon=":material/thumb_up:", duration="short")
    st.cache_data.clear()

    # Remember this paper's ID so we can keep its expander open
    st.session_state.last_expanded = paper_id

if df.empty:
    st.error("No dataset found!")
    st.stop()

st.sidebar.header("सौरभ का?")
password = st.sidebar.text_input("Enter password to edit", type="password")
if password == st.secrets["admin_password"]:
    st.sidebar.success("ये बाळा, वाच जरा!")
    st.subheader("Editing allowed")
else:
    st.subheader("My arXiv filtering tool")
    st.info(f"**Curated Daily arXiv Feed**\n\nAn automated classifier that scores daily arXiv papers in 'quant-ph + hep-lat + nucl-th' lists from 0 to 1 based on their alignment with my research interests. Because keeping up with 50+ abstracts a day should not be a full-time job.\n\nThe most recent entry in the database is from **{max_date.strftime('%m/%d/%Y')}**.", icon=":material/for_you:")

########################################################################
# If Passpord is correct
########################################################################

with st.sidebar:
    st.header("Filter")

    st.markdown("Show")
    show_read = st.sidebar.toggle(":green-badge[:material/auto_stories: Read]", value=True)
    show_unread = st.sidebar.toggle(":red-badge[:material/book: Unread]", value=True)

    show_star = st.sidebar.toggle(":violet-badge[:material/star: Highlighted]", value=True)
    show_unstar = st.sidebar.toggle(":yellow-badge[:material/circle: Others]", value=True)

    
    # Date Range
    default_start_date = max(min_date, max_date - pd.Timedelta(days=7))
    cal_max_day = pd.Timestamp.today().date()
    st.markdown("Select dates")
    date_selection = st.date_input(
        "Date Range",
        value=(default_start_date, max_date),
        min_value=min_date,
        max_value=cal_max_day,
        label_visibility="collapsed"
    )
    if len(date_selection) == 2:
        start_date, end_date = date_selection
        date_string = f" between {start_date.strftime('%m/%d/%Y')}-{end_date.strftime('%m/%d/%Y')}"
    elif len(date_selection) == 1:
        start_date = date_selection[0]
        end_date = date_selection[0]
        date_string = f" from {start_date.strftime('%m/%d/%Y')}"
    else:
        start_date, end_date = default_start_date, max_date
        date_string = ""

    st.markdown("Select a score range")
    min_score, max_score = st.sidebar.slider(
        "Select Score Range",
        min_value=0.8,
        max_value=1.0,
        value=(0.9, 1.0),  
        step=0.005,
        format="%0.3f",
        label_visibility="collapsed"
    )

    st.sidebar.subheader("Sorting")
    st.markdown("Sort by")
    sort_option = st.sidebar.selectbox(
        "Sort Data By",
        options=[
            "Score: Highest to Lowest",
            "Score: Lowest to Highest",
            "Date: Newest to Oldest",
            "Date: Oldest to Newest"
        ],
        label_visibility="collapsed"
    )

# 4. FILTERING LOGIC
# ---------------------------------------------------------
read_statuses = []
star_statuses=[]
if show_read: read_statuses.append(1)
if show_unread: read_statuses.append(0)
if show_star: star_statuses.append(1)
if show_unstar: star_statuses.append(0)
mask = (df['date_only'] >= start_date) & \
    (df['date_only'] <= end_date) & \
    (df['score'] >= min_score) & \
    (df['score'] <= max_score) &\
    (df['read'].isin(read_statuses)) &\
    (df['star'].isin(star_statuses))

filtered_df = df[mask]
total_matches = len(filtered_df)

if sort_option == "Score: Highest to Lowest":
    sort_cols = ['score']
    asc_opts = [False] 
elif sort_option == "Score: Lowest to Highest":
    sort_cols = ['score']
    asc_opts = [True]   
elif sort_option == "Date: Newest to Oldest":
    sort_cols = ['date_only']
    asc_opts = [False]  
elif sort_option == "Date: Oldest to Newest":
    sort_cols = ['date_only']
    asc_opts = [True]

filtered_df = filtered_df.sort_values(by=sort_cols, ascending=asc_opts)

# 5. PAGINATION CONTROLS
# ---------------------------------------------------------
PAGE_SIZE = 20
total_pages = math.ceil(total_matches / PAGE_SIZE)

# Ensure valid page number
if 'page' not in st.session_state:
    st.session_state.page = 1

with st.sidebar:
    st.write(f"**Matches:** {total_matches}")
    

# 6. DISPLAY LOOP
# ---------------------------------------------------------
if total_pages > 1:
    # Number input is better than slider for paging
    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, width=150)
    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
else:
    start_idx = 0
    end_idx = PAGE_SIZE

if total_matches>0:
    st.markdown(f"##### Showing {start_idx + 1}-{min(end_idx, total_matches)} of {total_matches}"+date_string)
else:
    st.info(f"**No matches found**\n\nThe database only has entries above 0.8 alignment score. Change the date range or score filter.", icon=":material/error:")

df_view = filtered_df.iloc[start_idx:end_idx]

for idx, row in df_view.iterrows():
    score_color = "green" if row['score'] > 0.9 else "orange"
    is_read = bool(row['read'])
    if is_read:
        expander_title = f"[{row['score']:.4f}] | :green-badge[:material/auto_stories:] | :green[{row['title']}]"
    else:
        expander_title = f"[{row['score']:.4f}] | :red-badge[:material/book:] | :red[{row['title']}]" 

    is_currently_expanded = st.session_state.get('last_expanded') == row['id']
    with st.expander(expander_title, expanded=is_currently_expanded):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**Score:** :{score_color}[{row['score']:.4f}]")
            st.markdown(f"**arXiv:** [{row['id']}](https://arxiv.org/abs/{row['id']}) [:blue-badge[:material/article: PDF]](https://arxiv.org/pdf/{row['id']})")
            st.markdown(f"**V1 on:** {row['date_only']}")
# ########################################################################
# # If the password is CORRECT
# ########################################################################
            if password == st.secrets["admin_password"]:
                st.checkbox(
                ":green-badge[:material/auto_stories: Mark read]",
                    value=bool(row['read']),
                    key=f"chk_{row['id']}",       
                    on_change=toggle_read,        
                    args=(row['id'],)             
                )
                st.checkbox(
                ":violet-badge[:material/star: Highlight]",
                    value=bool(row['star']),
                    key=f"star_{row['id']}",     
                    on_change=toggle_star,      
                    args=(row['id'],)          
                )
# ########################################################################
# # If the password is INCORRECT
# ########################################################################
            else:
                st.checkbox(
                ":green-badge[:material/auto_stories: Mark read]",
                    value=bool(row['read']),
                    key=f"chk_{row['id']}",        
                    args=(row['id'],),
                    disabled=True             
                )
                st.checkbox(
                ":violet-badge[:material/star: Highlight]",
                    value=bool(row['star']),
                    key=f"star_{row['id']}",        
                    args=(row['id'],),
                    disabled=True              
                )


        with col2:
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown(f"**Abstract:** {row['abstract']}")
