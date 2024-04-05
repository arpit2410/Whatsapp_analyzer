import nltk
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import nltk
from matplotlib import patheffects
nltk.download('vader_lexicon')
st.sidebar.title("Whatsapp Chat Analyzer ")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Which user ", user_list)

    if st.sidebar.button("Show Analysis"):
        col1, col2, col3, col4, col5 = st.columns(5)
        num_msgs, words, media_count, urlcount, emojilen = helper.fetchmsg(selected_user, df)
        with col1:
            st.header("Total Messages ")
            st.title(num_msgs)
        with col2:
            st.header("Total words")
            st.title(words)
        with col3:
            st.header("Total medias")
            st.title(media_count)
        with col4:
            st.header("URL counts")
            st.title(urlcount)
        with col5:
            st.header("Emoji length")
            st.title(emojilen)
        st.title("Sentiment Analysis: ")
        col1, col2, col3 = st.columns(3)
        data = helper.sentiments(selected_user, df)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Positive)</h3>",
                        unsafe_allow_html=True)
            d1 = data[data['value'] == 1]
            dcount = d1['month'].value_counts()
            fig7, ax = plt.subplots()
            ax.bar(dcount.index, dcount.values,
                   color=['teal', 'gold', 'skyblue', 'green', 'orange', 'red', 'cyan', 'lime'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig7)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)
            d2 = data[data['value'] == 0]
            dcount = d2['month'].value_counts()
            fig8, ax = plt.subplots()
            ax.bar(dcount.index, dcount.values,
                   color=['teal', 'gold', 'skyblue', 'green', 'orange', 'red', 'cyan', 'lime'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig8)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)
            d3 = data[data['value'] == -1]
            dcount = d3['month'].value_counts()
            fig8, ax = plt.subplots()
            ax.bar(dcount.index, dcount.values,
                   color=['teal', 'gold', 'skyblue', 'green', 'orange', 'red', 'cyan', 'lime'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig8)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity map(Positive)</h3>",
                        unsafe_allow_html=True)
            d1 = data[data['value'] == 1]
            dcount = d1['day_name'].value_counts()
            fig7, ax = plt.subplots()
            ax.bar(dcount.index, dcount.values, color=['red', 'gold', 'skyblue', 'green', 'orange'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig7)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)
            d2 = data[data['value'] == 0]
            dcount = d2['day_name'].value_counts()
            fig8, ax = plt.subplots()
            ax.bar(dcount.index, dcount.values, color=['red', 'gold', 'skyblue', 'green', 'orange'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig8)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)
            d3 = data[data['value'] == -1]
            dcount = d3['day_name'].value_counts()
            fig8, ax = plt.subplots()
            ax.bar(dcount.index, dcount.values, color=['red', 'gold', 'skyblue', 'green', 'orange'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig8)
        st.title("Percentage contributed")
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = helper.percentage(data, 1)
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = helper.percentage(data, 0)
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = helper.percentage(data, -1)
                st.dataframe(z)
            col1, col2, col3 = st.columns(3)
            x = data['user'][data['value'] == 1].value_counts().head(5)
            y = data['user'][data['value'] == -1].value_counts().head(5)
            z = data['user'][data['value'] == 0].value_counts().head(5)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'> Positive Users</h3>",
                            unsafe_allow_html=True)
                fig4, ax1 = plt.subplots()
                ax1.pie(x.values, labels=x.index,
                        colors=['yellow', 'violet', 'pink', 'skyblue', 'teal'], autopct='%1.1f%%', startangle=180,
                        pctdistance=0.85)
                st.pyplot(fig4)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Neutrals Users</h3>",
                            unsafe_allow_html=True)
                fig4, ax1 = plt.subplots()
                ax1.pie(y.values, labels=y.index,
                        colors=['yellow', 'violet', 'pink', 'skyblue', 'teal'], autopct='%1.1f%%', startangle=180,
                        pctdistance=0.85)
                st.pyplot(fig4)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Negative Users</h3>",
                            unsafe_allow_html=True)
                fig4, ax1 = plt.subplots()
                ax1.pie(z.values, labels=z.index,
                        colors=['yellow', 'violet', 'pink', 'skyblue', 'teal'], autopct='%1.1f%%', startangle=180,
                        pctdistance=0.85)
                st.pyplot(fig4)
            if selected_user == 'Overall':
                x = data['user'][data['value'] == 1].value_counts().head(10)
                y = data['user'][data['value'] == -1].value_counts().head(10)
                z = data['user'][data['value'] == 0].value_counts().head(10)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Users</h3>",
                                unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values,
                           color=['teal', 'gold', 'skyblue', 'green', 'orange', 'red', 'cyan', 'lime'])
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Neutrals Users</h3>",
                                unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.bar(z.index, z.values,
                           color=['teal', 'gold', 'skyblue', 'green', 'orange', 'red', 'cyan', 'lime'])
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Users</h3>",
                                unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.bar(y.index, y.values,
                           color=['teal', 'gold', 'skyblue', 'green', 'orange', 'red', 'cyan', 'lime'])
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
        st.header("Chat Analysis: ")
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_user(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color=['red', 'gold', 'skyblue', 'green', 'orange'])
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        st.title("WordCloud:")

        # Generate WordCloud data
        wordcloud_image = helper.create_wordcloud(selected_user, df)

        # Create plot with enhanced styling
        fig, ax = plt.subplots()  # Set figure size for better readability
        ax.imshow(wordcloud_image, interpolation='bilinear')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.title("Most Common Words: ")
        col1, col2 = st.columns(2)

        most_df = helper.most_common_word(selected_user, df)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(most_df[0],most_df[1], color=['red', 'gold', 'skyblue', 'green', 'orange'])
            plt.xticks(rotation='vertical')
            st.pyplot()
        with col2:
            st.dataframe(most_df)

        fem= helper.emojihelper(selected_user,df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(fem)
        with col2:
            fig2, ax1 = plt.subplots()
            ax1.pie(fem[1].head(), labels=fem[0].head(), autopct="%0.2f", colors = ['teal','gold','skyblue','green','orange','red','cyan','lime','orangered'])
            ax1.axis('equal')
            st.pyplot(fig2)
        st.title("Most Users: ")
        muse = helper.user_chat(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            fig4, ax1 = plt.subplots()
            ax1.pie(muse["Counts"].head(), labels=muse["Users"].head(),
                    colors=['red', 'gold', 'skyblue', 'green', 'pink'], autopct='%1.1f%%', startangle=180,
                    pctdistance=0.85)
            inner_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig4.gca().add_artist(inner_circle)
            st.pyplot(fig4)
        with col2:
            st.dataframe(muse)
        st.title("Most Active Weeks: ")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Busy Day")
            bday = helper.active_week(selected_user,df)
            fig5, ax = plt.subplots()
            ax.bar(bday["Day"], bday["Counts"], color=['red', 'gold', 'skyblue', 'green', 'pink'])
            st.pyplot(fig5)
        with col2:
            st.header("Weekly")
            st.dataframe(bday)
        st.title("Most Active Months: ")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Busy Month")
            bmonth = helper.active_month(selected_user, df)
            fig5, ax = plt.subplots()
            ax.bar(bmonth["Month"], bmonth["Counts"], color=['red', 'gold', 'skyblue', 'green', 'pink'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig5)
        with col2:
            st.header("Monthly")
            st.dataframe(bmonth)

                # col1, col2, col3 = st.columns(3)
                # temp = df[df['user'] != 'group_notification']
                # with col1:
                #     most_common_df = helper.most_senti(selected_user, data, 1)
                #     fig, ax = plt.subplots()
                #     ax.barh(most_common_df[0], most_common_df[1], color='green')
                #     plt.xticks(rotation='vertical')
                #     st.pyplot(fig)

    st.title("Monthly timeline: ")

    # Generate timeline data
    timeline = helper.timeline(selected_user, df)

    # Create plot with shadow and coloring
    fig2, ax = plt.subplots()
    ax.plot(timeline['time'].head(12), timeline['messages'].head(12), color='gray', linewidth=8, alpha=0.4)
    ax.plot(timeline['time'].head(12), timeline['messages'].head(12), color='blue', linewidth=4)
    ax.stackplot(timeline['time'].head(12), timeline['messages'].head(12), colors=['skyblue'])
    plt.xticks(rotation='vertical')
    st.pyplot(fig2)

    st.title("Daily timeline: ")
    dtimeline = helper.daily_timeline(selected_user, df)
    fig3, ax = plt.subplots()
    ax.plot(dtimeline['only_date'], dtimeline['messages'], color='green')

    plt.xticks(rotation='vertical')
    st.pyplot(fig3)

