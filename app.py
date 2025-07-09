import streamlit as st
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

# MUST be the first Streamlit command
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

def is_relevant_message(text, scheme):
    if not isinstance(text, str) or not text.strip():
        return False
    # Remove bot/moderator patterns and generic phrases
    bot_patterns = [
        r'i am a bot', r'this action was performed automatically', r'please contact the moderators',
        r'^if your image', r'^screenshots of social media', r'^as an outsider', r'^no shit sherlock',
        r'^how is this an unpopular opinion', r'^waiting for', r'^your post title should',
        r'^/r/india is not a substitute', r'^for relationship queries', r'^looks like a whatsapp forward',
        r'^at this point, why not', r'^at least they have', r'^i am fairly sure',
        r'^\^i\'m \^a \^bot', r'^\^op \^can \^reply', r'^\^message \^creator', r'^\^source',
        r'^\^did \^i \^just', r'^\^see \^how \^you', r'^visit \^the \^source', r'^here are some other news items',
        r'^self-posts als#rule', r'^one of the basic objectives', r'^country\'s largest bank',
        r'^the government has recently', r'^the finance ministry had earlier', r'^the ministry in its statement',
        r'^deposits in accounts', r'^accounts opened under', r'^the goals of', r'^pahal allowed people',
        r'^https?://', r'^\^', r'^$', r'^\s+$'
    ]
    for pat in bot_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return False
    # Must mention the scheme or a key word from the scheme
    scheme_keywords = [w.lower() for w in scheme.split() if len(w) > 3]
    if not any(kw in text.lower() for kw in scheme_keywords):
        return False
    # Remove messages that are too short or mostly non-alphabetic
    if len(text.strip()) < 30 or sum(c.isalpha() for c in text) < 20:
        return False
    return True

def get_sentiment_score(text, sentiment_label):
    """
    Calculate a more intuitive sentiment score based on positive/negative keywords
    """
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    
    # Positive sentiment keywords
    positive_keywords = [
        'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'good', 'beneficial', 
        'helpful', 'success', 'effective', 'useful', 'appreciate', 'love', 'like',
        'thank', 'grateful', 'positive', 'happy', 'satisfied', 'impressed', 'approve'
    ]
    
    # Negative sentiment keywords
    negative_keywords = [
        'bad', 'terrible', 'awful', 'useless', 'failed', 'failure', 'waste', 'corruption', 
        'scam', 'fraud', 'disappointed', 'hate', 'disgusted', 'angry', 'frustrated',
        'worst', 'pathetic', 'ridiculous', 'stupid', 'nonsense', 'fake', 'lie'
    ]
    
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    if sentiment_label.upper() == 'POSITIVE':
        return positive_count * 2 + len(text) / 100  # Bonus for positive keywords and length
    else:
        return negative_count * 2 + len(text) / 100  # Bonus for negative keywords and length

# Set a creative, colorful style for all plots
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    'axes.facecolor': '#f0f4f8',
    'figure.facecolor': '#f0f4f8',
    'axes.edgecolor': '#22223b',
    'axes.labelcolor': '#22223b',
    'xtick.color': '#22223b',
    'ytick.color': '#22223b',
    'grid.color': '#b5ead7',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
    'axes.titlecolor': '#3a86ff',
    'axes.labelsize': 13,
    'legend.frameon': True,
    'legend.facecolor': '#f7fff7',
    'legend.edgecolor': '#22223b',
})

# MongoDB connection
MONGO_URL = "mongodb+srv://kiran:0jf3f5IEAe8szOS1@govtsenti.yaqgau8.mongodb.net/"
def standardize_sentiment_labels_and_scores():
    """
    Standardize sentiment_label to uppercase and ensure sentiment_score exists in all collections.
    """
    mongo_url = MONGO_URL
    client = MongoClient(mongo_url)
    db = client['government_schemes']
    collections = ['cleaned_news', 'cleaned_reddit']
    for col in collections:
        collection = db[col]
        # Update 'positive' to 'POSITIVE'
        collection.update_many({'sentiment_label': 'positive'}, {'$set': {'sentiment_label': 'POSITIVE'}})
        # Update 'negative' to 'NEGATIVE'
        collection.update_many({'sentiment_label': 'negative'}, {'$set': {'sentiment_label': 'NEGATIVE'}})
        # Update 'neutral' to 'NEUTRAL'
        collection.update_many({'sentiment_label': 'neutral'}, {'$set': {'sentiment_label': 'NEUTRAL'}})
    print("✅ All sentiment labels have been standardized to uppercase.")

# Call this function once at the start (or expose as a Streamlit button if desired)
standardize_sentiment_labels_and_scores()

def check_and_fix_data_issues():
    """
    Check for data issues and fix common problems that might prevent records from showing.
    """
    mongo_url = MONGO_URL
    client = MongoClient(mongo_url)
    db = client['government_schemes']
    collections = ['cleaned_news', 'cleaned_reddit']
    
    for col in collections:
        collection = db[col]
        print(f"\n--- Checking {col} ---")
        
        # Check for null or empty scheme names
        null_schemes = collection.count_documents({'scheme_name': {'$in': [None, '', ' ']}})
        print(f"Records with null/empty scheme_name: {null_schemes}")
        
        # Check for null or empty sentiment data
        null_sentiment = collection.count_documents({
            '$or': [
                {'sentiment_label': {'$in': [None, '', ' ']}},
                {'sentiment_score': {'$in': [None, '', ' ']}}
            ]
        })
        print(f"Records with null/empty sentiment data: {null_sentiment}")
        
        # Check for National Education Policy variations
        nep_variations = [
            'National Education Policy',
            'national education policy',
            'NEP',
            'nep',
            'National education policy',
            'NATIONAL EDUCATION POLICY'
        ]
        
        # Exclude unwanted schemes from data check
        schemes_to_exclude = [
            'One Nation One Ration Card',
            'PM Mudra Yojana', 
            'Rural Digital Connectivity Scheme',
            'eShram Portal'
        ]
        
        total_nep = 0
        for variation in nep_variations:
            count = collection.count_documents({'scheme_name': variation})
            if count > 0:
                print(f"Found {count} records with scheme_name: '{variation}'")
                total_nep += count
        
        print(f"Total National Education Policy records: {total_nep}")
        print(f"Note: Excluded schemes from analysis: {', '.join(schemes_to_exclude)}")
        
        # Show sample records for debugging (excluding unwanted schemes)
        sample = list(collection.find({'scheme_name': {'$nin': schemes_to_exclude}}).limit(3))
        print(f"Sample records structure: {[{k: v for k, v in doc.items() if k != '_id'} for doc in sample]}")

# Add a button to run data check
# if st.sidebar.button("🔍 Check Data Issues"):
#     check_and_fix_data_issues()

@st.cache_data
def load_sentiment_data(mongo_url):
    client = MongoClient(mongo_url)
    db = client['government_schemes']
    dfs = []
    for collection_name, platform in [
        ('cleaned_news', 'News'),
        ('cleaned_reddit', 'Reddit')
    ]:
        # Debug: Check total records in collection
        total_records = db[collection_name].count_documents({})
        print(f"Total records in {collection_name}: {total_records}")
        
        # Debug: Check records with sentiment data
        sentiment_records = db[collection_name].count_documents(
            {'sentiment_label': {'$exists': True}, 'sentiment_score': {'$exists': True}}
        )
        print(f"Records with sentiment data in {collection_name}: {sentiment_records}")
        
        # Debug: Check National Education Policy records specifically
        nep_records = db[collection_name].count_documents(
            {'scheme_name': {'$regex': 'National Education Policy', '$options': 'i'}}
        )
        print(f"National Education Policy records in {collection_name}: {nep_records}")
        
        # Also check for variations in scheme name
        nep_variations = [
            'National Education Policy',
            'national education policy', 
            'NEP',
            'nep'
        ]
        for variation in nep_variations:
            var_count = db[collection_name].count_documents({'scheme_name': variation})
            if var_count > 0:
                print(f"Found {var_count} records with scheme_name: '{variation}' in {collection_name}")
        
        # Check all unique scheme names in this collection
        all_schemes = db[collection_name].distinct('scheme_name')
        print(f"All unique scheme names in {collection_name}: {all_schemes}")
        
        # More comprehensive query to catch all records
        # Different field names for different platforms: News uses 'description', Reddit uses 'comment'
        text_field = 'description' if platform == 'News' else 'comment'
        projection = {text_field: 1, 'scheme_name': 1, 'sentiment_label': 1, 'sentiment_score': 1, 'timestamp': 1, '_id': 0}
        
        data = list(db[collection_name].find(
            {
                'sentiment_label': {'$exists': True, '$ne': None, '$ne': '', '$nin': ['null', 'undefined']},
                'sentiment_score': {'$exists': True, '$ne': None, '$ne': '', '$nin': ['null', 'undefined']},
                'scheme_name': {'$exists': True, '$ne': None, '$ne': '', '$nin': ['null', 'undefined']}
            },
            projection
        ))
        
        print(f"Retrieved {len(data)} records from {collection_name}")
        
        if data:
            df = pd.DataFrame(data)
            df['platform'] = platform
            
            # Standardize text field name to 'description' for consistency
            text_field = 'description' if platform == 'News' else 'comment'
            if text_field in df.columns:
                if text_field != 'description':
                    df['description'] = df[text_field]
                    df = df.drop(columns=[text_field])
            
            # Clean and standardize scheme names
            df['scheme_name'] = df['scheme_name'].astype(str).str.strip()
            
            # Standardize all scheme name variations
            scheme_standardizations = {
                # National Education Policy variations
                'national education policy': 'National Education Policy',
                'National education policy': 'National Education Policy', 
                'NATIONAL EDUCATION POLICY': 'National Education Policy',
                'NEP': 'National Education Policy',
                'nep': 'National Education Policy',
                
                # Jan Aushadhi variations
                'jan aushadhi': 'Jan Aushadhi',
                'Jan aushadhi': 'Jan Aushadhi',
                'JAN AUSHADHI': 'Jan Aushadhi',
                'Pradhan Mantri Bhartiya Janaushadhi Pariyojana': 'Jan Aushadhi',
                'PMBJP': 'Jan Aushadhi',
                
                # PM-KISAN variations
                'pm-kisan': 'PM-KISAN',
                'PM-Kisan': 'PM-KISAN',
                'pm kisan': 'PM-KISAN',
                'PM KISAN': 'PM-KISAN',
                'Pradhan Mantri Kisan Samman Nidhi': 'PM-KISAN',
                
                # Ayushman Bharat variations
                'ayushman bharat': 'Ayushman Bharat',
                'AYUSHMAN BHARAT': 'Ayushman Bharat',
                'Ayushman bharat': 'Ayushman Bharat',
                'AB-PMJAY': 'Ayushman Bharat',
                'PMJAY': 'Ayushman Bharat',
                
                # Swachh Bharat Mission variations
                'swachh bharat mission': 'Swachh Bharat Mission',
                'SWACHH BHARAT MISSION': 'Swachh Bharat Mission',
                'Swachh bharat mission': 'Swachh Bharat Mission',
                'SBM': 'Swachh Bharat Mission',
                
                # Digital India variations
                'digital india': 'Digital India',
                'DIGITAL INDIA': 'Digital India',
                'Digital india': 'Digital India',
                
                # Make in India variations
                'make in india': 'Make in India',
                'MAKE IN INDIA': 'Make in India',
                'Make in india': 'Make in India',
                'Make In India': 'Make in India'
            }
            
            # Apply standardizations
            for old_name, new_name in scheme_standardizations.items():
                df.loc[df['scheme_name'].str.contains(old_name, case=False, na=False), 'scheme_name'] = new_name
            
            # Remove unwanted schemes at the platform level too
            schemes_to_remove = [
                'One Nation One Ration Card',
                'PM Mudra Yojana', 
                'Rural Digital Connectivity Scheme',
                'eShram Portal',
                'one nation one ration card',
                'pm mudra yojana',
                'rural digital connectivity scheme',
                'eshram portal'
            ]
            
            # Filter out unwanted schemes (case-insensitive)
            df = df[~df['scheme_name'].str.lower().isin([s.lower() for s in schemes_to_remove])]
            
            # Debug: Print unique scheme names for this platform
            print(f"Unique schemes in {platform} (after filtering): {df['scheme_name'].unique()}")
            print(f"Scheme counts in {platform}:")
            for scheme in df['scheme_name'].unique():
                count = (df['scheme_name'] == scheme).sum()
                print(f"  - {scheme}: {count}")
            dfs.append(df)
    
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        # Ensure all sentiment labels are uppercase in the DataFrame
        all_df['sentiment_label'] = all_df['sentiment_label'].str.upper()
        
        # Remove specific schemes that should not be displayed
        schemes_to_remove = [
            'One Nation One Ration Card',
            'PM Mudra Yojana', 
            'Rural Digital Connectivity Scheme',
            'eShram Portal'
        ]
        
        # Filter out the unwanted schemes
        all_df = all_df[~all_df['scheme_name'].isin(schemes_to_remove)]
        
        # Debug: Print final scheme counts after filtering
        print("Final scheme counts after removing unwanted schemes:")
        print(all_df['scheme_name'].value_counts())
        print(f"Total records loaded: {len(all_df)}")
        print(f"Schemes removed: {schemes_to_remove}")
        
        return all_df
    else:
        return pd.DataFrame()

st.title("\U0001F4CA Government Schemes Sentiment Analysis (News, Reddit)")

# Load data
all_df = load_sentiment_data(MONGO_URL)

if all_df.empty:
    st.warning("No sentiment analysis results found in MongoDB.")
    st.stop()

# Sidebar filters - user friendly with selectbox for single, multiselect for multiple
st.sidebar.header("Filter Options")

# Only show platforms and schemes that actually exist in the data
platforms = sorted(all_df['platform'].unique())
schemes = sorted([scheme for scheme in all_df['scheme_name'].dropna().unique() if scheme.strip() != ''])
sentiments = sorted(all_df['sentiment_label'].dropna().unique())

# Display available data counts in sidebar
st.sidebar.markdown("### Available Data:")
st.sidebar.write(f"**Total Records:** {len(all_df)}")
st.sidebar.write(f"**Platforms:** {len(platforms)}")
st.sidebar.write(f"**Schemes:** {len(schemes)}")
st.sidebar.write(f"**Sentiments:** {len(sentiments)}")

# Show scheme distribution in sidebar
st.sidebar.markdown("### Scheme Distribution:")
scheme_counts = all_df['scheme_name'].value_counts()
for scheme, count in scheme_counts.items():
    st.sidebar.write(f"- {scheme}: {count}")

selected_platform = st.sidebar.selectbox("Platform", ["All"] + platforms)
selected_scheme = st.sidebar.selectbox("Scheme", ["All"] + schemes)
selected_sentiments = st.sidebar.multiselect("Sentiment(s)", sentiments, default=sentiments)

# Filter data
filtered_df = all_df.copy()
if selected_platform != "All":
    filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
if selected_scheme != "All":
    filtered_df = filtered_df[filtered_df['scheme_name'] == selected_scheme]
if selected_sentiments:
    filtered_df = filtered_df[filtered_df['sentiment_label'].isin(selected_sentiments)]

st.markdown(f"### Showing {len(filtered_df)} results")

# Data Table
st.dataframe(filtered_df, use_container_width=True)

# Download Button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='sentiment_results.csv',
    mime='text/csv'
)

# Colorful palette for all graphs
color_palette = sns.color_palette("Set2")

# Visualizations
# Only show 'Sentiment Distribution by Platform' if not filtered to a single platform
if selected_platform == "All":
    st.subheader("Sentiment Distribution by Platform")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # Ensure all platforms are shown even if they have zero counts
    all_platforms = ['News', 'Reddit']
    sns.countplot(data=filtered_df, x='sentiment_label', hue='platform', palette=color_palette, ax=ax1, 
                  hue_order=all_platforms)
    ax1.set_title("Sentiment Distribution by Platform")
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Number of Messages")
    ax1.set_facecolor('#f0f4f8')
    plt.tight_layout()
    st.pyplot(fig1)
    # Insights for this graph
    total_msgs = len(filtered_df)
    plat_counts = filtered_df['platform'].value_counts()
    most_msgs_platform = plat_counts.idxmax() if not plat_counts.empty else 'N/A'
    most_msgs_count = plat_counts.max() if not plat_counts.empty else 0
    most_sentiment = filtered_df['sentiment_label'].value_counts().idxmax() if not filtered_df.empty else 'N/A'
    st.info(f"""
    - The above bar chart shows the distribution of sentiment labels across all platforms.
    - {most_msgs_platform} has the highest number of messages ({most_msgs_count}).
    - The most common sentiment overall is '{most_sentiment}'.
    - Use the filters to focus on a specific platform or sentiment.
    """)
    # Pie chart for sentiment distribution by platform
    st.markdown("**Pie Chart: Sentiment Proportion by Platform**")
    pie_data = filtered_df.groupby('platform')['sentiment_label'].value_counts().unstack().fillna(0)
    # Ensure all platforms are included even if they have zero counts
    all_platforms = ['News', 'Reddit']
    available_platforms = [p for p in all_platforms if p in pie_data.index]
    
    for plat in available_platforms:
        if pie_data.loc[plat].sum() > 0:  # Only create pie chart if there's data
            fig_pie, ax_pie = plt.subplots()
            pie_data.loc[plat].plot.pie(autopct='%1.1f%%', colors=color_palette, ax=ax_pie, startangle=90)
            ax_pie.set_ylabel('')
            ax_pie.set_title(f"{plat} Sentiment Proportion")
            st.pyplot(fig_pie)
            # Insights for pie chart
            plat_total = pie_data.loc[plat].sum()
            top_sent = pie_data.loc[plat].idxmax()
            top_sent_count = pie_data.loc[plat].max()
            st.info(f"""
            - This pie chart shows the sentiment breakdown for {plat}.
            - The most common sentiment is '{top_sent}' ({top_sent_count} out of {plat_total} messages).
            - The distribution helps identify platform-specific sentiment trends.
            """)

# Only show 'Sentiment Distribution per Scheme' if not filtered to a single scheme
if selected_scheme == "All":
    st.subheader("Sentiment Distribution per Scheme (All Platforms)")
    
    # Check if we have any schemes to display
    available_schemes = filtered_df['scheme_name'].dropna().unique()
    if len(available_schemes) > 0:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.countplot(data=filtered_df, x='scheme_name', hue='sentiment_label', palette=color_palette, ax=ax2)
        ax2.set_title("Sentiment Distribution per Scheme")
        ax2.set_xlabel("Government Scheme")
        ax2.set_ylabel("Number of Messages")
        ax2.set_facecolor('#f0f4f8')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Insights for this graph
        scheme_counts = filtered_df['scheme_name'].value_counts()
        top_scheme = scheme_counts.idxmax() if not scheme_counts.empty else 'N/A'
        top_scheme_count = scheme_counts.max() if not scheme_counts.empty else 0
        st.info(f"""
        - The above chart shows sentiment distribution for each government scheme.
        - '{top_scheme}' has the highest number of messages ({top_scheme_count}).
        - Compare sentiment trends across schemes to identify which schemes have more positive or negative coverage.
        - Use the filters to focus on a specific scheme or sentiment.
        """)
        
        # Pie chart for sentiment distribution per scheme
        st.markdown("**Pie Chart: Sentiment Proportion per Scheme**")
        pie_scheme = filtered_df.groupby('scheme_name')['sentiment_label'].value_counts().unstack().fillna(0)
        for sch in pie_scheme.index:
            if pie_scheme.loc[sch].sum() > 0:  # Only create pie chart if there's data
                fig_pie2, ax_pie2 = plt.subplots()
                pie_scheme.loc[sch].plot.pie(autopct='%1.1f%%', colors=color_palette, ax=ax_pie2, startangle=90)
                ax_pie2.set_ylabel('')
                ax_pie2.set_title(f"{sch} Sentiment Proportion")
                st.pyplot(fig_pie2)
                # Insights for pie chart
                sch_total = pie_scheme.loc[sch].sum()
                if sch_total > 0:
                    top_sent = pie_scheme.loc[sch].idxmax()
                    top_sent_count = pie_scheme.loc[sch].max()
                    st.info(f"""
                    - This pie chart shows the sentiment breakdown for {sch}.
                    - The most common sentiment is '{top_sent}' ({top_sent_count} out of {sch_total} messages).
                    - This helps identify which schemes have more positive or negative sentiment overall.
                    """)
    else:
        st.warning("No schemes found in the filtered data to display.")

# Only show 'Platform-wise Sentiment per Scheme' if not filtered to a single platform or single scheme
if selected_platform == "All" and selected_scheme == "All":
    st.subheader("Platform-wise Sentiment per Scheme")
    
    # Get available schemes that have data
    available_schemes = filtered_df['scheme_name'].dropna().unique()
    if len(available_schemes) > 0:
        for scheme in available_schemes:
            scheme_df = filtered_df[filtered_df['scheme_name'] == scheme]
            
            # Only show scheme if it has data
            if len(scheme_df) > 0:
                st.markdown(f"**{scheme}**")
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                
                # Get available platforms for this scheme
                available_platforms_for_scheme = scheme_df['platform'].unique()
                all_platforms = ['News', 'Reddit']
                
                # Use available platforms order, but ensure consistent order
                platform_order = [p for p in all_platforms if p in available_platforms_for_scheme]
                
                if len(platform_order) > 0:
                    sns.countplot(data=scheme_df, x='platform', hue='sentiment_label', 
                                palette=color_palette, ax=ax3, order=platform_order)
                    
                    ax3.set_title(f"Sentiment for '{scheme}' by Platform")
                    ax3.set_xlabel("Platform")
                    ax3.set_ylabel("Number of Messages")
                    ax3.set_facecolor('#f0f4f8')
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    # Insights for this graph
                    plat_counts = scheme_df['platform'].value_counts()
                    if not plat_counts.empty:
                        top_platform = plat_counts.idxmax()
                        top_platform_count = plat_counts.max()
                    else:
                        top_platform = 'N/A'
                        top_platform_count = 0
                    
                    st.info(f"""
                    - This chart shows how sentiment for '{scheme}' varies across platforms.
                    - {top_platform} has the most messages for this scheme ({top_platform_count}).
                    - Available platforms for this scheme: {', '.join(available_platforms_for_scheme)}.
                    - Use filters to focus on a specific platform or sentiment for deeper insights.
                    """)
                    
                    # Pie chart for each scheme's sentiment by platform
                    st.markdown(f"**Pie Chart: {scheme} Sentiment by Platform**")
                    pie_scheme_platform = scheme_df.groupby('platform')['sentiment_label'].value_counts().unstack().fillna(0)
                    
                    # Ensure we show pie charts for available platforms with data
                    available_platforms = pie_scheme_platform.index.tolist()
                    for plat in available_platforms:
                        if pie_scheme_platform.loc[plat].sum() > 0:  # Only create pie chart if there's data
                            fig_pie3, ax_pie3 = plt.subplots()
                            pie_scheme_platform.loc[plat].plot.pie(autopct='%1.1f%%', colors=color_palette, ax=ax_pie3, startangle=90)
                            ax_pie3.set_ylabel('')
                            ax_pie3.set_title(f"{scheme} - {plat} Sentiment Proportion")
                            st.pyplot(fig_pie3)
                            # Insights for pie chart
                            plat_total = pie_scheme_platform.loc[plat].sum()
                            if plat_total > 0:
                                top_sent = pie_scheme_platform.loc[plat].idxmax()
                                top_sent_count = pie_scheme_platform.loc[plat].max()
                                st.info(f"""
                                - This pie chart shows the sentiment breakdown for {scheme} on {plat}.
                                - The most common sentiment is '{top_sent}' ({top_sent_count} out of {plat_total} messages).
                                - This helps compare how the same scheme is perceived on different platforms.
                                """)
                else:
                    st.warning(f"No platform data available for {scheme}")
    else:
        st.warning("No schemes available to display platform-wise sentiment.")

# --- Overall Sentiment Combining Reddit and News ---
news_reddit_df = all_df[all_df['platform'].isin(['News', 'Reddit'])]
if not news_reddit_df.empty:
    st.subheader("Overall Sentiment: News + Reddit")
    fig_overall, ax_overall = plt.subplots()
    sns.countplot(data=news_reddit_df, x='sentiment_label', palette=color_palette, ax=ax_overall)
    ax_overall.set_title("Combined Sentiment Distribution (News & Reddit)")
    ax_overall.set_xlabel("Sentiment")
    ax_overall.set_ylabel("Number of Messages")
    ax_overall.set_facecolor('#f0f4f8')
    st.pyplot(fig_overall)
    # Insights for overall sentiment
    total_msgs = len(news_reddit_df)
    most_sentiment = news_reddit_df['sentiment_label'].value_counts().idxmax() if not news_reddit_df.empty else 'N/A'
    most_sentiment_count = news_reddit_df['sentiment_label'].value_counts().max() if not news_reddit_df.empty else 0
    st.info(f"""
    - This chart shows the overall sentiment distribution by combining News and Reddit data.
    - The most common sentiment is '{most_sentiment}' ({most_sentiment_count} out of {total_msgs} messages).
    - This helps you understand the general public and media perception across both platforms.
    - Use this to compare with platform-specific or scheme-specific trends above.
    """)

# --- Show Top Positive/Negative Messages for Each Scheme (News + Reddit Combined) ---
st.subheader("Top Positive/Negative Messages by Scheme (News + Reddit Combined)")
news_reddit_df = filtered_df[filtered_df['platform'].isin(['News', 'Reddit'])]

# Get available schemes that have data
available_schemes = news_reddit_df['scheme_name'].dropna().unique()
if len(available_schemes) > 0:
    for scheme in available_schemes:
        scheme_df = news_reddit_df[news_reddit_df['scheme_name'] == scheme]
        
        # Only proceed if we have data for this scheme
        if len(scheme_df) > 0:
            st.markdown(f"### {scheme}")
            col1, col2 = st.columns(2)
            
            # Top 5 Positive messages with clear sentiment indicators
            pos_msgs = scheme_df[scheme_df['sentiment_label'].str.upper() == 'POSITIVE']
            pos_msgs = pos_msgs[pos_msgs['description'].apply(lambda x: is_relevant_message(x, scheme))]
            
            if not pos_msgs.empty:
                # Calculate intuitive sentiment scores
                pos_msgs = pos_msgs.copy()
                pos_msgs['intuitive_score'] = pos_msgs['description'].apply(
                    lambda x: get_sentiment_score(x, 'POSITIVE')
                )
                
                # Get top 5 based on intuitive score
                top_pos = pos_msgs.nlargest(5, 'intuitive_score')
                
                with col1:
                    st.markdown("**Top 5 Most Positive Messages:**")
                    if len(top_pos) > 0:
                        for i, row in top_pos.iterrows():
                            original_score = row.get('sentiment_score', 'N/A')
                            platform = row.get('platform', 'Unknown')
                            description = row['description']
                            
                            # Highlight positive keywords in the text
                            positive_keywords = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                                               'good', 'beneficial', 'helpful', 'success', 'effective', 
                                               'useful', 'appreciate', 'love', 'like', 'thank']
                            
                            # Truncate and show message
                            display_text = description[:250] + "..." if len(description) > 250 else description
                            
                            st.write(f"**{platform}** - {display_text}")
                            
                            # Show original sentiment score for reference
                            if original_score != 'N/A':
                                st.caption(f"Original Score: {original_score:.3f}")
                            st.write("---")
                    else:
                        st.write("No clear positive messages found for this scheme.")
            else:
                with col1:
                    st.markdown("**Top 5 Most Positive Messages:**")
                    st.write("No positive messages found for this scheme.")
            
            # Top 5 Negative messages with clear sentiment indicators
            neg_msgs = scheme_df[scheme_df['sentiment_label'].str.upper() == 'NEGATIVE']
            neg_msgs = neg_msgs[neg_msgs['description'].apply(lambda x: is_relevant_message(x, scheme))]
            
            if not neg_msgs.empty:
                # Calculate intuitive sentiment scores
                neg_msgs = neg_msgs.copy()
                neg_msgs['intuitive_score'] = neg_msgs['description'].apply(
                    lambda x: get_sentiment_score(x, 'NEGATIVE')
                )
                
                # Get top 5 based on intuitive score
                top_neg = neg_msgs.nlargest(5, 'intuitive_score')
                
                with col2:
                    st.markdown("**Top 5 Most Negative Messages:**")
                    if len(top_neg) > 0:
                        for i, row in top_neg.iterrows():
                            original_score = row.get('sentiment_score', 'N/A')
                            platform = row.get('platform', 'Unknown')
                            description = row['description']
                            
                            # Truncate and show message
                            display_text = description[:250] + "..." if len(description) > 250 else description
                            
                            st.write(f"**{platform}** - {display_text}")
                            
                            # Show original sentiment score for reference
                            if original_score != 'N/A':
                                st.caption(f"Original Score: {original_score:.3f}")
                            st.write("---")
                    else:
                        st.write("No clear negative messages found for this scheme.")
            else:
                with col2:
                    st.markdown("**Top 5 Most Negative Messages:**")
                    st.write("No negative messages found for this scheme.")
                    
            # Add summary insights for each scheme
            total_pos = len(scheme_df[scheme_df['sentiment_label'].str.upper() == 'POSITIVE'])
            total_neg = len(scheme_df[scheme_df['sentiment_label'].str.upper() == 'NEGATIVE'])
            
            st.info(f"""
            **{scheme} Summary:**
            - Positive messages: {total_pos}
            - Negative messages: {total_neg}
            - The messages above are selected based on clear sentiment indicators to help you understand public perception.
            """)
else:
    st.warning("No schemes available to display top messages.")

# --- Word Clouds for Each Sentiment and Scheme ---
st.subheader("Word Clouds by Sentiment and Scheme")

# Get available schemes that have data
available_schemes = filtered_df['scheme_name'].dropna().unique()
if len(available_schemes) > 0:
    for scheme in available_schemes:
        scheme_df = filtered_df[filtered_df['scheme_name'] == scheme]
        
        # Only proceed if we have data for this scheme
        if len(scheme_df) > 0:
            st.markdown(f"### {scheme}")
            available_sentiments = scheme_df['sentiment_label'].dropna().unique()
            
            for sentiment in available_sentiments:
                sent_df = scheme_df[scheme_df['sentiment_label'] == sentiment]
                if len(sent_df) > 0:
                    # Combine all descriptions for this sentiment and scheme
                    descriptions = sent_df['description'].dropna().astype(str)
                    text = ' '.join(descriptions)
                    
                    # Clean and check if we have meaningful text
                    cleaned_text = text.strip()
                    word_count = len(cleaned_text.split()) if cleaned_text else 0
                    
                    # Generate word cloud if we have at least 3 words and 20 characters
                    if cleaned_text and word_count >= 3 and len(cleaned_text) >= 20:
                        st.markdown(f"**Sentiment: {sentiment}** ({len(sent_df)} messages)")
                        try:
                            # Adjust max_words based on available text
                            max_words = min(50, word_count) if word_count < 100 else 100
                            wordcloud = WordCloud(width=600, height=300, background_color='white', 
                                                colormap='Set2', max_words=max_words, 
                                                min_word_length=2).generate(cleaned_text)
                            st.image(wordcloud.to_array(), use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate word cloud for {scheme} - {sentiment}: {str(e)}")
                    elif cleaned_text and word_count > 0:
                        # Show basic info if we have some text but not enough for word cloud
                        st.markdown(f"**Sentiment: {sentiment}** ({len(sent_df)} messages)")
                        st.info(f"Only {word_count} words available - insufficient for word cloud generation")
                    else:
                        # Only show this if there's truly no text data
                        st.markdown(f"**Sentiment: {sentiment}** ({len(sent_df)} messages)")
                        st.info("No text data available for word cloud generation")
else:
    st.warning("No schemes available to display word clouds.")