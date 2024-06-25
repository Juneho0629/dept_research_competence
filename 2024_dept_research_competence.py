import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import mysql.connector

st.set_page_config(layout="wide")

streamlit_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nanum+Square:wght@400;700&display=swap');
body, div, dl, dt, dd, ul, ol, li, h1, h2, h3, h4, h5, h6, pre, form, p, blockquote, th, td {
    font-family: 'NanumSquare', sans-serif;
}
</style>
"""

connection = mysql.connector.connect(
    host = st.secrets['mysql']['host'],
    user = st.secrets['mysql']['user'],
    password = st.secrets['mysql']['password'],
    database = st.secrets['mysql']['database']
)

st.markdown(streamlit_style, unsafe_allow_html=True)

def main():
    # 페이지 제목
    st.header("학과별 연구역량(논문) 분석보고서")

    try:
        with connection.cursor() as cursor:
            table_name = "department_scival_2019_2023"
            cursor.execute(f"SELECT * FROM {table_name}")
            result = cursor.fetchall()

            # 컬럼명 가져오기
            column_names = [desc[0] for desc in cursor.description]

            # 데이터프레임으로 변환
            df = pd.DataFrame(result, columns=column_names)

            category = df['계열'].unique()
            selected_category = st.selectbox("계열 선택", category)

            dept = df[df['계열'] == selected_category]['학과분류'].unique()
            selected_dept = st.selectbox("학과 선택", dept)
            selected_data = df[df['학과분류'] == selected_dept]  

            table_name = selected_dept
            cursor.execute(f"SELECT * FROM {table_name}")
            result = cursor.fetchall()

            # 컬럼명 가져오기
            column_names = [desc[0] for desc in cursor.description]

            # 데이터프레임으로 변환
            univ_comparison = pd.DataFrame(result, columns=column_names)

            # 논문 수, 논문 1편당 피인용수, 평균 FWCI, FWCI 상위10% 논문비율, SNIP기준 상위 10% 저널 게재비율, 국제공동연구 논문 비율을 계산합니다.
            univ_comparison['total_citations'] = univ_comparison[['citation_count_2019', 'citation_count_2020', 'citation_count_2021', 'citation_count_2022', 'citation_count_2023']].sum(axis=1)
            univ_comparison['논문 수'] = univ_comparison.groupby('대학교명')['eid'].transform('count')
            univ_comparison['논문 1편당 피인용수'] = round(univ_comparison.groupby('대학교명')['total_citations'].transform('mean'), 1)
            univ_comparison['평균 FWCI'] = round(univ_comparison.groupby('대학교명')['FieldWeightedCitationImpact'].transform('mean'),2)
            univ_comparison['FWCI 상위 10% 논문비율(%)'] = round(univ_comparison.groupby('대학교명')['OutputsInTopCitationPercentiles_10'].transform('sum') / univ_comparison['논문 수'] * 100, 1)
            univ_comparison['SNIP기준 상위 10% 저널 게재비율(%)'] = round(univ_comparison.groupby('대학교명')['PublicationsInTopJournalPercentiles_10'].transform('sum') / univ_comparison['논문 수'] * 100, 1)
            univ_comparison['국제공동연구 논문 비율(%)'] = round(univ_comparison.groupby('대학교명')['InternationalCollaboration'].transform('sum') / univ_comparison['논문 수'] * 100, 1)
            univ_comparison['Year'] = univ_comparison['prism:coverDate'].str[:4]

            # 중복된 행을 제거하고 필요한 칼럼만 선택하여 comparison 데이터프레임을 생성합니다.
            comparison = univ_comparison.drop_duplicates(subset=['대학교명'])[['대학교명', '논문 수', '논문 1편당 피인용수', '평균 FWCI', 'FWCI 상위 10% 논문비율(%)', 'SNIP기준 상위 10% 저널 게재비율(%)', '국제공동연구 논문 비율(%)']]
            comparison['교원 수'] = comparison.대학교명.map(selected_data.groupby('대학교명')['성명'].count())
            comparison['교원 1인당 연간 논문수'] = round((comparison['논문 수'] / comparison['교원 수']) / 5, 1)
            comparison['피인용수'] = comparison.대학교명.map(univ_comparison.groupby('대학교명')['total_citations'].sum())
            comparison = comparison[['대학교명', '교원 수', '논문 수', '교원 1인당 연간 논문수', '피인용수', '논문 1편당 피인용수', '평균 FWCI', 'FWCI 상위 10% 논문비율(%)', 'SNIP기준 상위 10% 저널 게재비율(%)', '국제공동연구 논문 비율(%)']]

            # # 비율 칼럼들을 %로 변환합니다.
            # percentage_columns = ['FWCI 상위10% 논문비율', 'SNIP기준 상위 10% 저널 게재비율', '국제공동연구 논문 비율']
            # for col in percentage_columns:
            #     comparison[col] = comparison[col].apply(lambda x: f'{x}%')

            # '연세대학교'를 가장 위로 올립니다.
            comparison = comparison.set_index('대학교명')
            comparison = comparison.reindex(['연세대학교'] + [index for index in comparison.index if index != '연세대학교']).reset_index()

            st.markdown(f"※ 비교 대학: {', '.join(selected_data.대학교명.unique())}")
            
            st.divider()

            st.subheader('일러두기')
            st.markdown('**1. 분석대상 연구자: 2024. 3. 1. 기준**')
            st.markdown('&#160;&#160;&#160;가. 연세대: 교무처에서 제공받은 전임교원 명단 기준')
            st.markdown('&#160;&#160;&#160;나. 타대학: 학과 홈페이지에 공개된 전임교원 명단 기준')
            st.markdown('**2. 분석대상 논문: 2018-2022에 게재된 Scopus 등재 논문**')
            st.markdown('&#160;&#160;&#160;저자의 Affiliation이 연세대가 아닌 논문 포함')
            st.markdown('**3. 분석 도구: SciVal(Scopus 등재논문 분석)**')
            st.markdown('**4. 분석의 제한사항**')
            st.markdown('&#160;&#160;&#160;가. Scopus에서 구분한 연구자별 author id를 기준으로 논문을 분석하였으므로 연구자별 실제 업적과는 차이가 있을 수도 있음')
            st.markdown('&#160;&#160;&#160;나. Scopus에 해당 연구자의 author id가 여러개 존재할 경우, KRI에 등재된 해당 연구자 논문 실적과 학과 홈페이지에 공개된 CV를 참고하여 가장 유사한 author id를 선택하여 분석하였음')
            st.markdown('&#160;&#160;&#160;다. 타대학 전임교원의 경우 홈페이지에 최신정보 반영이 늦을 수 있기 때문에 대상 연구자가 실제와 다를 수도 있음')

            st.divider()

            st.subheader(f'I. 연세대학교 {selected_dept} 논문성과')
            st.subheader('(2019-2023)')
            st.markdown('**1. 개요**')
            st.dataframe(comparison.iloc[:1,:], use_container_width=True, hide_index=True)

            st.markdown('**2. [양적지표] 논문 수 및 연구분야**')
            st.markdown('&#160;&#160;&#160;**가. 연도별 논문 수, 교원 1인당 논문 수**')

            # 연세대학교의 데이터를 필터링하고, 년도별 논문 수 계산
            yonsei_scholarly_outputs = univ_comparison[univ_comparison['대학교명'] == '연세대학교'].groupby('Year')['eid'].size().to_frame().T

            # 총 논문 수 계산 및 데이터프레임에 추가
            yonsei_scholarly_outputs['계'] = yonsei_scholarly_outputs.sum(axis=1)

            # 컬럼 순서 재정렬 ('계'를 맨 앞으로 이동)
            columns = ['계'] + [col for col in yonsei_scholarly_outputs.columns if col != '계']
            yonsei_scholarly_outputs = yonsei_scholarly_outputs[columns]

            # 컬럼명을 '논문 수'로 변경
            yonsei_scholarly_outputs.index = ['논문 수']

            # 2번째 행 추가 (1번째 행을 21로 나눈 값)
            yonsei_scholarly_outputs.loc['교원 1인당 논문 수'] = round(yonsei_scholarly_outputs.loc['논문 수'] / selected_data[selected_data.대학교명 == '연세대학교']['성명'].count(),1)
            yonsei_scholarly_outputs['계']['교원 1인당 논문 수'] = yonsei_scholarly_outputs['계']['교원 1인당 논문 수'] / 5

            st.dataframe(yonsei_scholarly_outputs, use_container_width=True)
            
            # 연도별 논문 수와 교원 1인당 논문 수 데이터 추출
            논문수_data = yonsei_scholarly_outputs.loc['논문 수'][['2019','2020','2021','2022','2023']]
            교원1인당논문수_data = yonsei_scholarly_outputs.loc['교원 1인당 논문 수'][['2019','2020','2021','2022','2023']]

            # Streamlit columns 생성
            col1, col2 = st.columns(2)

            # 첫 번째 column에 '논문 수' bar 그래프 생성
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=논문수_data.index, y=논문수_data.values, width=0.4, marker_color='lightskyblue'))
                fig1.add_shape(type="line", x0=논문수_data.index[0], y0=np.mean(논문수_data.values), x1=논문수_data.index[-1], y1=np.mean(논문수_data.values), line=dict(color="orangered", width=2, dash="dash"))
                fig1.update_layout(title_text='연도별 논문 수', annotations=[go.layout.Annotation(text=f"평균: {np.mean(논문수_data.values):.1f}", showarrow=False, xref="paper", yref="y", x=1, y=np.mean(논문수_data.values)+np.std(논문수_data.values))])
                fig1.update_traces(text=논문수_data.values, textposition='auto')
                st.plotly_chart(fig1)

            # 두 번째 column에 '교원 1인당 논문 수' bar 그래프 생성
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=교원1인당논문수_data.index, y=교원1인당논문수_data.values, width=0.4, marker_color='lightskyblue'))
                fig2.add_shape(type="line", x0=교원1인당논문수_data.index[0], y0=np.mean(교원1인당논문수_data.values), x1=교원1인당논문수_data.index[-1], y1=np.mean(교원1인당논문수_data.values), line=dict(color="orangered", width=2, dash="dash"))
                fig2.update_layout(title_text='연도별 교원 1인당 논문 수', annotations=[go.layout.Annotation(text=f"평균: {np.mean(교원1인당논문수_data.values):.1f}", showarrow=False, xref="paper", yref="y", x=1, y=np.mean(교원1인당논문수_data.values)+np.std(교원1인당논문수_data.values))])
                fig2.update_traces(text=교원1인당논문수_data.values, textposition='auto')
                st.plotly_chart(fig2)

            st.markdown('&#160;&#160;&#160;**나. 연구주제별 현황**')

            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            학과에서 생산된 논문들이 여러 학문 분야 중 어느 분야에 주로 포함되어 있는지 살펴봄으로써, 해당 주제분야에서 학과의 논문은 얼마나 영향력이 있는지, 그리고 해당 주제분야가 타 주제분야에 비해 성장하는 분야인지 쇠퇴하고 있는 분야인지 등 주제분야 단위의 분석을 가능하게 합니다.
            (참고) Scopus는 학문단위 분석을 위해 학문분야를 분류하고 이를 저널에 할당하고 있습니다.
            현재 27개 Subject Area와 그 하위에 334개 Subject Field로 분류하여, 이를 ASJC(All Science Journal Classification)라 명명하고, Scopus의 모든 저널을 334개의 Subject Field 중 1개 이상의 영역에 포함시키고 있습니다. 
            * 출처: https://service.elsevier.com/app/answers/detail/a_id/15181/supporthub/scopus/
            </div>
            """, unsafe_allow_html=True)

            # 연세대학교 데이터 추출
            yonsei = univ_comparison[univ_comparison['대학교명'] == '연세대학교']
            yonsei_ASJC = yonsei.assign(SubjectAreas=yonsei['SubjectAreas'].str.split(';')).explode('SubjectAreas')

            # 'SubjectAreas'별로 '논문 수', '비율', '피인용수', '논문당 피인용수', 'FWCI' 계산
            ASJC_논문수 = yonsei_ASJC['SubjectAreas'].str.strip().value_counts()
            ASJC_비율 = round((yonsei_ASJC['SubjectAreas'].str.strip().value_counts() / ASJC_논문수.sum())*100,1)
            ASJC_피인용수 = yonsei_ASJC.groupby(yonsei_ASJC['SubjectAreas'].str.strip())[['citation_count_2019', 'citation_count_2020', 'citation_count_2021', 'citation_count_2022', 'citation_count_2023']].sum().sum(axis=1)
            ASJC_논문당피인용수 = round(ASJC_피인용수 / ASJC_논문수,1)
            ASJC_FWCI = round(yonsei_ASJC.groupby(yonsei_ASJC['SubjectAreas'].str.strip())['FieldWeightedCitationImpact'].mean(),2)

            # 새로운 데이터프레임 생성
            df_ASJC = pd.DataFrame({
                '논문 수': ASJC_논문수,
                '비율(%)': ASJC_비율,
                '피인용수': ASJC_피인용수,
                '논문당 피인용수': ASJC_논문당피인용수,
                'FWCI': ASJC_FWCI
            })

            # '논문 수' 내림차순으로 정렬
            df_ASJC = df_ASJC.sort_values(by='논문 수', ascending=False)
            df_ASJC.index.name = 'Subject Area'
            
            # 상위 6개 항목 추출
            top10 = df_ASJC.nlargest(10, '비율(%)')

            # 나머지 항목을 'Others'로 합침
            others = df_ASJC.drop(top10.index)
            others_row = others.sum(numeric_only=True)
            others_row.name = 'Others'
            others_df = pd.DataFrame([others_row])  # others_row를 DataFrame으로 변환
            combined = pd.concat([top10, others_df])  # DataFrame에 concat

            donut_colors = px.colors.qualitative.Set3_r

            # 도넛차트 생성
            fig3 = go.Figure(data=[go.Pie(labels=combined.index, values=combined['비율(%)'], hole=.3)])
            fig3.update_layout(height=700, 
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=-0.2,
                                    xanchor='right',
                                    x=0.5
                                ))
            st.plotly_chart(fig3, use_container_width=True)

            # 상위 6개 요소를 갖고 양쪽을 마주보는 bar chart 생성
            categories = top10.index.tolist()[::-1]

            fig4 = go.Figure()

            # 왼쪽 bar chart (논문 수)
            fig4.add_trace(go.Bar(
                y=categories,
                x=-top10.reindex(categories)['논문 수'],  # 왼쪽으로 향하도록 음수로 설정
                orientation='h',
                name='논문 수',
                text=top10.reindex(categories)['논문 수'],
                textposition='outside',  # 텍스트 위치 조정
                marker=dict(color='palegoldenrod'),
                hovertemplate='논문 수: %{text}',  # hovertemplate을 사용하여 호버링 정보 지정
                hoverinfo='text',
                width=0.5
            ))

            # 오른쪽 bar chart (FWCI)
            fig4.add_trace(go.Bar(
                y=categories,
                x=top10.reindex(categories)['FWCI'] * (top10['논문 수'].max() / top10['FWCI'].max()),
                orientation='h',
                name='FWCI',
                text=top10.reindex(categories)['FWCI'],
                textposition='outside',
                marker=dict(color='lightskyblue'),
                hovertemplate='FWCI: %{text}',  # hovertemplate을 사용하여 호버링 정보 지정
                hoverinfo='text',
                width=0.5
            ))

            # 레이아웃 업데이트
            fig4.update_layout(
                barmode='overlay',
                xaxis=dict(
                    title='논문 수    FWCI',
                    tickvals=[],  # x축 눈금 숨기기
                    ticktext=[],  # x축 눈금 텍스트 숨기기
                    showgrid=False  # 격자선 숨기기
                ),
                xaxis2=dict(
                    overlaying='x',
                    side='top',
                    title='FWCI (스케일 조정됨)',
                    tickvals=[0, top10['논문 수'].max()],
                    ticktext=['0', f'{top10["논문 수"].max()}']
                ),
                yaxis=dict(
                    title='Subject Areas'
                ),
                title='<주요 연구주제별 논문의 FWCI>',
                showlegend=True,
                height=600
            )

            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(df_ASJC, use_container_width=True)

            st.markdown('&#160;&#160;&#160;**다. Topic Cluster**')

            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            Topic은 공통의 'intellecutual interest'를 갖는 Scopus 등재 논문들의 집합이며, Topic Cluster는 서로 관련있는 Topic들의 집합체입니다. Scopus 등재논문들간의 '직접적 인용분석'을 통하여, Topic 및 Topic Cluster가 생성되며, 현재 96,000개의 Topic과 1,500개의 Topic Cluster가 생성되어있습니다.
            Subject Classification은 저널의 주제를 기준으로 인위적으로 분류되어 생성되는 반면, Topic / Topic Cluster는 개별 논문의 citation analysis를 기반으로 자연적으로 생성되어 지속되기도 하며 사라지기도 합니다. 즉, 특정 저널에 게재된 모든 논문은 동일한 Subject Classification에 포함되지만, 다양한 Topic/Topic Cluster에 포함됩니다.
            하나의 논문은 하나의 Topic에 속하며, 하나의 Topic은 하나의 Topic Cluster에 생성됩니다. 특정 Topic/Topic Cluster에 포함된 논문집합을 분석함으로써, 해당 Topic/Topic Cluster의 모멘텀을 확인할 수 있으며, 이 지표를 Prominence percentile이라고 합니다. 이 지표는 citation count, view count, CiteScore를 바탕으로 계산되며, 100%에 가까울 수록 지속적으로 발전가능하며 향후 연구투자가 활발히 일어날 수 있는 분야입니다. Prominence percentile과 funding과의 상관관계가 높다는 것은 여러 보고서에서 입증되었습니다.
            아래의 지표를 통해 현재 학과의 논문이 어떤 Topic/Topic Cluster에 많이 포함되어있는지, 해당 Topic의 모멘텀은 어떤지 확인할 수 있습니다.
            * 출처: https://service.elsevier.com/app/answers/detail/a_id/28428/supporthub/scival/
            </div>
            """, unsafe_allow_html=True)

            # 'TCPP'별로 '논문 수', '비율', '피인용수', '논문당 피인용수', 'FWCI' 계산
            TCPP_논문수 = yonsei.groupby('topicClusterName').size()
            ASJC_FWCI = round(yonsei.groupby('topicClusterName')['FieldWeightedCitationImpact'].mean(),2)
            ASJC_PP = round(yonsei.groupby('topicClusterName')['prominencePercentile'].mean(),2)

            # 새로운 데이터프레임 생성
            df_TCPP = pd.DataFrame({
                '논문 수': TCPP_논문수,
                'FWCI': ASJC_FWCI,
                'Prominence Percentile': ASJC_PP
            })

            df_TCPP.index.name = 'Topic Cluster'
            # '논문 수' 내림차순으로 정렬
            df_TCPP = df_TCPP.sort_values(by='논문 수', ascending=False)
            st.dataframe(df_TCPP, use_container_width=True)

            st.markdown('**3. [질적지표] 피인용수를 기반으로 한 지표**')
            st.markdown('&#160;&#160;&#160;**가. 피인용수, FWCI**')

            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            FWCI(Field-Weighted Citation Impact)는 피인용 수치에 영향을 미치는 변수인 연구분야, Document type, 게재년도(노출기간) 등을 정규화하여, 세계 평균값인 "1.0" 대비 피인용 비율을 측정하는 상대적 인용영향력입니다. 예를 들어 FWCI 지수가 1.34이면, 전세계 평균대비 34% 더 인용되었다고 해석 가능합니다.
            * 출처: https://service.elsevier.com/app/answers/detail/a_id/28192/supporthub/scival/p/10961/28192/
            </div>
            """, unsafe_allow_html=True)

            df_fwci = pd.DataFrame({
                '피인용 수': yonsei.groupby('Year')['total_citations'].sum(),
                '논문당 피인용 수': yonsei.groupby('Year')['total_citations'].sum() / yonsei.groupby('Year')['eid'].count(),
                'FWCI': yonsei.groupby('Year')['FieldWeightedCitationImpact'].mean()
            })
            df_fwci = df_fwci.T
            df_fwci['계'] = [df_fwci.loc['피인용 수'].sum(), yonsei.total_citations.sum() / yonsei.eid.count(), yonsei.FieldWeightedCitationImpact.mean()]
            df_fwci.loc['논문당 피인용 수'] = df_fwci.loc['논문당 피인용 수'].round(1)
            df_fwci.loc['FWCI'] = df_fwci.loc['FWCI'].round(2)
            df_fwci = df_fwci[['계', '2019', '2020', '2021', '2022', '2023']]

            # 1행 2열로 그래프를 그리기 위해 두 개의 컬럼을 생성합니다.
            col1, col2 = st.columns(2)

            with col1:
                # 첫 번째 그래프를 그립니다. (사진의 왼쪽 그래프를 복제)
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(x=df_fwci.iloc[:,1:].columns, y=df_fwci.iloc[:,1:].loc['피인용 수'], name='피인용 수', text=df_fwci.iloc[:,1:].loc['피인용 수'], textposition='auto', marker_color='lightskyblue', width=0.5))
                fig5.add_trace(go.Scatter(x=df_fwci.columns, y=df_fwci.loc['논문당 피인용 수'], mode='lines+markers+text', name='논문당 피인용 수', yaxis='y2',
                                    marker=dict(symbol='circle', color='white', line=dict(color='brown', width=1), size=10, sizemode='diameter'),
                                    line=dict(color='brown'),
                                    text=df_fwci.loc['논문당 피인용 수'], textposition='top center', textfont=dict(color='brown')))
                # 보조축을 설정합니다.
                fig5.update_layout(
                    yaxis2=dict(
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        zeroline=False
                    )
                )
                st.plotly_chart(fig5)

            with col2:
                # 두 번째 그래프를 그립니다. (사진의 오른쪽 그래프를 복제)
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(x=df_fwci.columns, y=df_fwci.loc['FWCI'], mode='lines+markers+text', name='FWCI',
                                        marker=dict(symbol='circle', color='white', line=dict(color='royalblue', width=1), size=10, sizemode='diameter'),
                                        line=dict(color='royalblue'),
                                        text=df_fwci.loc['FWCI'], textposition='top center'))
                fig6.update_layout(xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']))
                st.plotly_chart(fig6)

            st.dataframe(df_fwci, use_container_width=True)

            st.markdown('&#160;&#160;&#160;**나. FWCI 상위 x% 논문 수**')

            outputintop = pd.DataFrame({
                'FWCI 상위 1% 논문수': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_1'].sum(),
                'FWCI 상위 1% 논문 비율': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_1'].sum() / yonsei.groupby('Year').size() * 100,
                'FWCI 상위 5% 논문수': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_5'].sum(),
                'FWCI 상위 5% 논문 비율': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_5'].sum() / yonsei.groupby('Year').size() * 100,
                'FWCI 상위 10% 논문수': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_10'].sum(),
                'FWCI 상위 10% 논문 비율': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_10'].sum() / yonsei.groupby('Year').size() * 100,
                'FWCI 상위 25% 논문수': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_25'].sum(),
                'FWCI 상위 25% 논문 비율': yonsei.groupby('Year')['OutputsInTopCitationPercentiles_25'].sum() / yonsei.groupby('Year').size() * 100
            })
            outputintop = outputintop.T
            outputintop = outputintop.round(1)

            outputintop['계'] = [outputintop.loc['FWCI 상위 1% 논문수'].sum(), outputintop.loc['FWCI 상위 1% 논문수'].sum()/ len(yonsei) * 100,
            outputintop.loc['FWCI 상위 5% 논문수'].sum(), outputintop.loc['FWCI 상위 5% 논문수'].sum()/ len(yonsei) * 100,
            outputintop.loc['FWCI 상위 10% 논문수'].sum(), outputintop.loc['FWCI 상위 10% 논문수'].sum()/ len(yonsei) * 100,
            outputintop.loc['FWCI 상위 25% 논문수'].sum(), outputintop.loc['FWCI 상위 25% 논문수'].sum()/ len(yonsei) * 100,
            ]
            outputintop['계'] = outputintop['계'].round(1)
            columns = ['계'] + [col for col in outputintop.columns if col != '계']
            outputintop = outputintop[columns]

            outputintop_percentile = outputintop.iloc[1::2,:].copy()
            outputintop_percentile.loc['FWCI 상위 5% 논문 비율'] = (outputintop.loc['FWCI 상위 5% 논문 비율'] - outputintop.loc['FWCI 상위 1% 논문 비율']).round(1)
            outputintop_percentile.loc['FWCI 상위 10% 논문 비율'] = (outputintop.loc['FWCI 상위 10% 논문 비율'] - outputintop.loc['FWCI 상위 5% 논문 비율']).round(1)
            outputintop_percentile.loc['FWCI 상위 25% 논문 비율'] = (outputintop.loc['FWCI 상위 25% 논문 비율'] - outputintop.loc['FWCI 상위 10% 논문 비율']).round(1)
            outputintop_percentile.loc['FWCI 상위 100% 논문 비율'] = (100 - outputintop.loc['FWCI 상위 25% 논문 비율']).round(1)

            # Transpose the dataframe and drop the first column for plotting.
            outputintop_percentile_transposed = outputintop_percentile.iloc[:,1:].T

            # Define the color for each category
            colors = {
                'FWCI 상위 1% 논문 비율': 'red',
                'FWCI 상위 5% 논문 비율': 'midnightblue',
                'FWCI 상위 10% 논문 비율': 'lightgreen',
                'FWCI 상위 25% 논문 비율': 'aliceblue',
                'FWCI 상위 100% 논문 비율': 'gainsboro'
            }

            # Create a stacked bar chart using Plotly.
            fig7 = go.Figure(data=[
                go.Bar(name=col, x=outputintop_percentile_transposed.index, y=outputintop_percentile_transposed[col], 
                        marker_color=colors[col], text=outputintop_percentile_transposed[col], textposition='auto',
                        width=0.4) for col in reversed(outputintop_percentile_transposed.columns)
            ])

            # Change the bar mode
            fig7.update_layout(barmode='stack', height=600)

            # Show the figure
            st.plotly_chart(fig7, use_container_width=True)

            st.dataframe(outputintop, use_container_width=True)

            st.markdown('&#160;&#160;&#160;**다. 상위 x% 저널에 게재된 논문 수(SNIP 기준)**')
            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            SNIP(Source Normalized Impact per Paper)는 저널의 주제분야가 고려되지 않은 저널의 영향력지수인 J.I.F.(Journal Impact Factor)와 달리 주제분야별 인용패턴이 고려된 저널의 영향력지수로써 평균값은 "1.0"입니다. 
            특정년도의 Journal Impact(최근 3년간의 논문 1편당 평균피인용 횟수)를 인용잠재력(RDCP, Relative Database Citation Potential)로 나누어 계산합니다.
            * 출처: https://www.journalindicators.com/methodology
            </div>
            """, unsafe_allow_html=True)

            pubintop = pd.DataFrame({
                '상위 1% 저널 게재논문 수': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_1'].sum(),
                '상위 1% 저널 게재논문 비율': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_1'].sum() / yonsei.groupby('Year').size() * 100,
                '상위 5% 저널 게재논문 수': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_5'].sum(),
                '상위 5% 저널 게재논문 비율': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_5'].sum() / yonsei.groupby('Year').size() * 100,
                '상위 10% 저널 게재논문 수': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_10'].sum(),
                '상위 10% 저널 게재논문 비율': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_10'].sum() / yonsei.groupby('Year').size() * 100,
                '상위 25% 저널 게재논문 수': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_25'].sum(),
                '상위 25% 저널 게재논문 비율': yonsei.groupby('Year')['PublicationsInTopJournalPercentiles_25'].sum() / yonsei.groupby('Year').size() * 100
            })
            pubintop = pubintop.T
            pubintop = pubintop.round(1)

            pubintop['계'] = [pubintop.loc['상위 1% 저널 게재논문 수'].sum(), pubintop.loc['상위 1% 저널 게재논문 수'].sum()/ len(yonsei) * 100,
            pubintop.loc['상위 5% 저널 게재논문 수'].sum(), pubintop.loc['상위 5% 저널 게재논문 수'].sum()/ len(yonsei) * 100,
            pubintop.loc['상위 10% 저널 게재논문 수'].sum(), pubintop.loc['상위 10% 저널 게재논문 수'].sum()/ len(yonsei) * 100,
            pubintop.loc['상위 25% 저널 게재논문 수'].sum(), pubintop.loc['상위 25% 저널 게재논문 수'].sum()/ len(yonsei) * 100,
            ]
            pubintop['계'] = pubintop['계'].round(1)
            columns = ['계'] + [col for col in pubintop.columns if col != '계']
            pubintop = pubintop[columns]

            pubintop_percentile = pubintop.iloc[1::2,:].copy()
            pubintop_percentile.loc['상위 5% 저널 게재논문 비율'] = (pubintop.loc['상위 5% 저널 게재논문 비율'] - pubintop.loc['상위 1% 저널 게재논문 비율']).round(1)
            pubintop_percentile.loc['상위 10% 저널 게재논문 비율'] = (pubintop.loc['상위 10% 저널 게재논문 비율'] - pubintop.loc['상위 5% 저널 게재논문 비율']).round(1)
            pubintop_percentile.loc['상위 25% 저널 게재논문 비율'] = (pubintop.loc['상위 25% 저널 게재논문 비율'] - pubintop.loc['상위 10% 저널 게재논문 비율']).round(1)
            pubintop_percentile.loc['상위 100% 저널 게재논문 비율'] = (100 - pubintop.loc['상위 25% 저널 게재논문 비율']).round(1)
            # Transpose the dataframe and drop the first column for plotting.
            pubintop_percentile_transposed = pubintop_percentile.iloc[:,1:].T

            # Define the color for each category
            colors = {
                '상위 1% 저널 게재논문 비율': 'red',
                '상위 5% 저널 게재논문 비율': 'midnightblue',
                '상위 10% 저널 게재논문 비율': 'lightgreen',
                '상위 25% 저널 게재논문 비율': 'aliceblue',
                '상위 100% 저널 게재논문 비율': 'gainsboro'
            }

            # Create a stacked bar chart using Plotly.
            fig8 = go.Figure(data=[
                go.Bar(name=col, x=pubintop_percentile_transposed.index, y=pubintop_percentile_transposed[col], 
                        marker_color=colors[col], text=pubintop_percentile_transposed[col], textposition='auto',
                        width=0.4) for col in reversed(pubintop_percentile_transposed.columns)
            ])

            # Change the bar mode
            fig8.update_layout(barmode='stack', height=600)

            # Show the figure
            st.plotly_chart(fig8, use_container_width=True)

            st.dataframe(pubintop, use_container_width=True)

            st.divider()

            anonymity_option = '무기명'

            # 성명 처리 함수
            def process_name(name, anonymity):
                if anonymity == '무기명':
                    return '*' * len(name)
                return name

            # 성명 처리
            selected_data['성명'] = selected_data['성명'].apply(lambda x: process_name(x, anonymity_option))

            st.subheader(f'II. {selected_dept} 교원별 논문성과')
            st.markdown('**1. 양적지표**')
            st.markdown('&#160;&#160;&#160;**가. 논문 수**')

            # 데이터를 내림차순으로 정렬합니다.
            sorted_output = selected_data.sort_values('Overall_ScholarlyOutput', ascending=False)
            sorted_output['성명_ID'] = sorted_output['성명'] + '_' + sorted_output['Author_ID'].astype(str)
            sorted_output['연번'] = range(1, len(sorted_output) + 1)

            # 평균을 구합니다.
            average = np.mean(sorted_output['Overall_ScholarlyOutput'])

            # Plotly bar 차트를 그립니다.
            fig1 = px.bar(sorted_output, x=sorted_output.성명_ID, y='Overall_ScholarlyOutput', 
                        labels={'Overall_ScholarlyOutput':'논문 수', '성명_ID':'연번'},
                        #title=f'{selected_dept} 교원별 논문성과(상위 50인)',
                        hover_data=['성명', '대학교명'],
                        color_discrete_sequence=['lightskyblue']
                        )

            # x축 레이블을 '연번'으로 표시하도록 업데이트합니다.
            fig1.update_xaxes(tickvals=sorted_output.성명_ID, ticktext=sorted_output.연번, tickangle=0)  
            fig1.update_layout(height=700)

            # horizontal dash line을 그립니다.
            fig1.add_shape(type="line", x0=0, x1=1, y0=average, y1=average, xref='paper', yref='y',
                        line=dict(color="orange", width=1, dash="dash"))

            # 평균 수치를 text로 추가합니다.
            fig1.add_annotation(x=len(sorted_output)/2, y=average, text=f"평균: {average:.2f}", showarrow=True, arrowhead=2, arrowcolor="orange", font=dict(size=14, color="darkorange"), align="center")

            # 평균 범례를 추가합니다.
            fig1.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="orange", width=1, dash="dash"), name="평균"))

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig1, use_container_width=True)

            # 2019-2023 데이터 처리
            for year in range(2019, 2024):
                selected_data[f'ScholarlyOutput_{year}'] = np.random.randint(0, 5, size=len(selected_data))

            sorted_output["논문 수('19~'23)"] = sorted_output[['ScholarlyOutput_2019', 'ScholarlyOutput_2020', 'ScholarlyOutput_2021', 'ScholarlyOutput_2022', 'ScholarlyOutput_2023']].apply(lambda x: '-'.join(x.astype(str)), axis=1)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df1 = sorted_output[['연번', '성명', '대학교명', 'Overall_ScholarlyOutput', "논문 수('19~'23)"]].copy()
            display_df1.columns = ['연번', '성명', '대학교', '계', "논문 수('19~'23)"]

            st.dataframe(display_df1, use_container_width=True, hide_index=True)

            st.markdown('**2. 질적지표**')
            st.markdown('&#160;&#160;&#160;**가. 논문 1편당 피인용수**')

            # 데이터를 내림차순으로 정렬합니다.
            sorted_citations_per_publications = selected_data.sort_values('Overall_CitationsPerPublication', ascending=False)
            sorted_citations_per_publications['성명_ID'] = sorted_citations_per_publications['성명'] + '_' + sorted_citations_per_publications['Author_ID'].astype(str)
            sorted_citations_per_publications['연번'] = range(1, len(sorted_citations_per_publications) + 1)

            # 평균을 구합니다.
            average = np.mean(sorted_citations_per_publications['Overall_CitationsPerPublication'])

            # Plotly bar 차트를 그립니다.
            fig2 = px.bar(sorted_citations_per_publications, x=sorted_citations_per_publications.성명_ID, y='Overall_CitationsPerPublication', 
                        labels={'Overall_CitationsPerPublication':'논문 1편당 피인용수', '성명_ID':'연번'},
                        #title=f'{selected_dept} 교원별 논문 1편당 피인용(상위 50인)',
                        hover_data=['성명', '대학교명'],
                        color_discrete_sequence=['lightskyblue']
                        )

            # x축 레이블을 '성명'만 표시하도록 업데이트합니다.
            fig2.update_xaxes(tickvals=sorted_citations_per_publications.성명_ID, ticktext=sorted_citations_per_publications.연번, tickangle=0)
            fig2.update_layout(height=700)

            # horizontal dash line을 그립니다.
            fig2.add_shape(type="line", x0=0, x1=1, y0=average, y1=average, xref='paper', yref='y',
                        line=dict(color="orange", width=1, dash="dash"))

            # 평균 수치를 text로 추가합니다.
            fig2.add_annotation(x=len(sorted_citations_per_publications)/2, y=average, text=f"평균: {average:.2f}", showarrow=True, arrowhead=2, arrowcolor="orange", font=dict(size=14, color="darkorange"), align="center")

            # 평균 범례를 추가합니다.
            fig2.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="orange", width=1, dash="dash"), name="평균"))

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig2, use_container_width=True)

            # '논문 수('19~'23)'에 해당하는 데이터를 다시 생성합니다.
            sorted_citations_per_publications["논문1편당 피인용수"] = round(sorted_citations_per_publications['Overall_CitationsPerPublication'], 1)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df2 = sorted_citations_per_publications[['연번', '성명', '대학교명', '논문1편당 피인용수']].copy()
            display_df2.columns = ['연번', '성명', '대학교', "논문1편당 피인용수"]

            st.dataframe(display_df2, use_container_width=True, hide_index=True)

            st.markdown('&#160;&#160;&#160;**나. FWCI**')

            # 데이터를 내림차순으로 정렬합니다.
            sorted_fwci = selected_data.sort_values('Overall_FieldWeightedCitationImpact', ascending=False)
            sorted_fwci['성명_ID'] = sorted_fwci['성명'] + '_' + sorted_fwci['Author_ID'].astype(str)
            sorted_fwci['연번'] = range(1, len(sorted_fwci) + 1)

            # 평균을 구합니다.
            average = np.mean(sorted_fwci['Overall_FieldWeightedCitationImpact'])

            # Plotly bar 차트를 그립니다.
            fig3 = px.bar(sorted_fwci, x=sorted_fwci.성명_ID, y='Overall_FieldWeightedCitationImpact', 
                        labels={'Overall_FieldWeightedCitationImpact':'FWCI', '성명_ID':'연번'},
                        #title=f'{selected_dept} 교원별 FWCI(상위 50인)',
                        hover_data=['성명', '대학교명'],
                        color_discrete_sequence=['lightskyblue']
                        )

            # x축 레이블을 '성명'만 표시하도록 업데이트합니다.
            fig3.update_xaxes(tickvals=sorted_fwci.성명_ID, ticktext=sorted_fwci.연번, tickangle=0)
            fig3.update_layout(height=700)

            # horizontal dash line을 그립니다.
            fig3.add_shape(type="line", x0=0, x1=1, y0=average, y1=average, xref='paper', yref='y',
                        line=dict(color="orange", width=1, dash="dash"))

            # 평균 수치를 text로 추가합니다.
            fig3.add_annotation(x=len(sorted_fwci)/2, y=average, text=f"평균: {average:.2f}", showarrow=True, arrowhead=2, arrowcolor="orange", font=dict(size=14, color="darkorange"), align="center")

            # 평균 범례를 추가합니다.
            fig3.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="orange", width=1, dash="dash"), name="평균"))

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig3, use_container_width=True)

            sorted_fwci["FWCI"] = round(sorted_fwci['Overall_FieldWeightedCitationImpact'], 1)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df3 = sorted_fwci[['연번', '성명', '대학교명', 'FWCI']].copy()
            display_df3.columns = ['연번', '성명', '대학교', "FWCI"]

            st.dataframe(display_df3, use_container_width=True, hide_index=True)

            st.markdown('&#160;&#160;&#160;**다. FWCI 상위 10% 이내 논문 수**')

            # 데이터를 내림차순으로 정렬합니다.
            sorted_outputs_in_top_10 = selected_data.sort_values('Overall_OutputsInTopCitationPercentiles_10', ascending=False)
            sorted_outputs_in_top_10['성명_ID'] = sorted_outputs_in_top_10['성명'] + '_' + sorted_outputs_in_top_10['Author_ID'].astype(str)
            sorted_outputs_in_top_10['연번'] = range(1, len(sorted_outputs_in_top_10) + 1)

            # 평균을 구합니다.
            average = np.mean(sorted_outputs_in_top_10['Overall_OutputsInTopCitationPercentiles_10'])

            # Plotly bar 차트를 그립니다.
            fig4 = px.bar(sorted_outputs_in_top_10, x=sorted_outputs_in_top_10.성명_ID, y='Overall_OutputsInTopCitationPercentiles_10', 
                        labels={'Overall_OutputsInTopCitationPercentiles_10':'FWCI 상위 10% 이내 논문 수', '성명_ID':'연번'},
                        #title=f'{selected_dept} 교원별 FWCI 상위 10% 이내 논문 수(상위 50인)',
                        hover_data=['성명', '대학교명'],
                        color_discrete_sequence=['lightskyblue']
                        )

            # x축 레이블을 '성명'만 표시하도록 업데이트합니다.
            fig4.update_xaxes(tickvals=sorted_outputs_in_top_10.성명_ID, ticktext=sorted_outputs_in_top_10.연번, tickangle=0)
            fig4.update_layout(height=700)

            # horizontal dash line을 그립니다.
            fig4.add_shape(type="line", x0=0, x1=1, y0=average, y1=average, xref='paper', yref='y',
                        line=dict(color="orange", width=1, dash="dash"))

            # 평균 수치를 text로 추가합니다.
            fig4.add_annotation(x=len(sorted_outputs_in_top_10)/2, y=average, text=f"평균: {average:.2f}", showarrow=True, arrowhead=2, arrowcolor="orange", font=dict(size=14, color="darkorange"), align="center")

            # 평균 범례를 추가합니다.
            fig4.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="orange", width=1, dash="dash"), name="평균"))

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig4, use_container_width=True)

            # '논문 수('19~'23)'에 해당하는 데이터를 다시 생성합니다.
            sorted_outputs_in_top_10["FWCI 상위 10% 이내 논문 수"] = round(sorted_outputs_in_top_10['Overall_OutputsInTopCitationPercentiles_10'], 1)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df4 = sorted_outputs_in_top_10[['연번', '성명', '대학교명', 'FWCI 상위 10% 이내 논문 수']].copy()
            display_df4.columns = ['연번', '성명', '대학교', "FWCI 상위 10% 이내 논문 수"]

            st.dataframe(display_df4, use_container_width=True, hide_index=True)

            st.markdown('&#160;&#160;&#160;**라. 논문 수-FWCI-FWCI 상위 10% 이내 논문 수**')
            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            세계대학평가에서는 대체적으로 FWCI가 높은 논문이 많을수록, 그리고 HCR에서는 citation이 높은 논문(HCPs, 피인용 상위 1%내 논문)수가 많을수록 우수한 연구자라고 평가하고 있습니다. 
            이를 한눈에 보기 위해 "논문 수-FWCI-FWCI 상위 10% 이내 논문 수"를 하나의 차트에 표현하였습니다. 버블의 크기가 크고, 오른쪽 상단에 위치하는 연구자가 많을수록 HCR 선정 및 세계대학평가에 유리하다고 할 수 있습니다.
            </div>
            """, unsafe_allow_html=True)

            # 데이터를 내림차순으로 정렬합니다.
            sorted_mix = selected_data.sort_values(by=['Overall_ScholarlyOutput', 'Overall_FieldWeightedCitationImpact', 'Overall_OutputsInTopCitationPercentiles_10'], ascending=False)
            sorted_mix['성명_ID'] = sorted_mix['성명'] + '_' + sorted_mix['Author_ID'].astype(str)
            sorted_mix['연번'] = range(1, len(sorted_mix) + 1)
            sorted_mix["FWCI"] = round(sorted_fwci['Overall_FieldWeightedCitationImpact'], 1)

            fig5 = px.scatter(sorted_mix, x='Overall_ScholarlyOutput', y='FWCI',  
                        labels={'Overall_ScholarlyOutput':'논문 수', 'Overall_OutputsInTopCitationPercentiles_10': 'FWCI 상위 10% 이내 논문 수'},
                        hover_data=['성명', '대학교명'],
                        size='Overall_OutputsInTopCitationPercentiles_10',  
                        color='Overall_OutputsInTopCitationPercentiles_10',  
                        color_continuous_scale='Turbo'
                        )

            fig5.update_layout(height=700)

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig5, use_container_width=True)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df5 = sorted_mix[['연번', '성명', '대학교명', 'Overall_ScholarlyOutput', 'FWCI', 'Overall_OutputsInTopCitationPercentiles_10']].copy()
            display_df5.columns = ['연번', '성명', '대학교', "논문 수", 'FWCI', 'FWCI 상위 10% 논문 수']

            st.dataframe(display_df5, use_container_width=True, hide_index=True)

            st.markdown('&#160;&#160;&#160;**마. H-index**')
            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            H-index는 연구의 양(논문 수)과 질(citation)을 동시에 나타내기 위한 지표로서, 연구자의 논문을 피인용빈도가 높은 순으로 나열하였을 때, 논문의 피인용수가 논문의 순위보다 크거나 같은 마지막 논문의 순위로 산정합니다. 
            예를 들어, 어느 연구자의 H-index가 10이라면, 연구자의 논문 10편은 최소 10회 이상의 피인용을 받았다는 것을 의미합니다.
            </div>
            """, unsafe_allow_html=True)

            # 데이터를 내림차순으로 정렬합니다.
            sorted_hindex = selected_data.sort_values('HIndices', ascending=False)
            sorted_hindex['성명_ID'] = sorted_hindex['성명'] + '_' + sorted_hindex['Author_ID'].astype(str)
            sorted_hindex['연번'] = range(1, len(sorted_hindex) + 1)

            # 평균을 구합니다.
            average = np.mean(sorted_hindex['HIndices'])

            # Plotly bar 차트를 그립니다.
            fig6 = px.bar(sorted_hindex, x=sorted_hindex.성명_ID, y='HIndices', 
                        labels={'HIndices':'H-index', '성명_ID':'연번'},
                        #title=f'{selected_dept} 교원별 H-index(상위 50인)',
                        hover_data=['성명', '대학교명'],
                        color_discrete_sequence=['lightskyblue']
                        )

            # x축 레이블을 '성명'만 표시하도록 업데이트합니다.
            fig6.update_xaxes(tickvals=sorted_hindex.성명_ID, ticktext=sorted_hindex.연번, tickangle=0)
            fig6.update_layout(height=700)

            # horizontal dash line을 그립니다.
            fig6.add_shape(type="line", x0=0, x1=1, y0=average, y1=average, xref='paper', yref='y',
                        line=dict(color="orange", width=1, dash="dash"))

            # 평균 수치를 text로 추가합니다.
            fig6.add_annotation(x=len(sorted_hindex)/2, y=average, text=f"평균: {average:.2f}", showarrow=True, arrowhead=2, arrowcolor="orange", font=dict(size=14, color="darkorange"), align="center")

            # 평균 범례를 추가합니다.
            fig6.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="orange", width=1, dash="dash"), name="평균"))

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig6, use_container_width=True)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df6 = sorted_hindex[['연번', '성명', '대학교명', 'HIndices']].copy()
            display_df6.columns = ['연번', '성명', '대학교', "H-index"]

            st.dataframe(display_df6, use_container_width=True, hide_index=True)

            st.markdown('**3. 공동연구**')
            st.markdown("""
            <div style="background-color: #FFFFE0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);">
            융복합연구를 통한 새로운 학문영역이 창출되면 이를 바탕으로 한 후속연구가 활발해지며, 공동연구를 수행하는 연구진 그룹 규모가 클수록 후속연구가 다양해집니다. 
            공동연구(특히 국제공동연구)에 의한 논문은 그렇지 않은 논문보다 피인용 관련지표(FWCI 등)가 높은 경향이 있다고 보고되면서 공동연구에 관한 지표가 주목받고 있습니다.
            </div>
            """, unsafe_allow_html=True)

            # 데이터를 내림차순으로 정렬합니다.
            sorted_colab = selected_data.sort_values('Collaboration_International collaboration_percentage', ascending=False)
            sorted_colab['성명_ID'] = sorted_colab['성명'] + '_' + sorted_colab['Author_ID'].astype(str)
            sorted_colab['연번'] = range(1, len(sorted_colab) + 1)
            sorted_colab['Collaboration_International collaboration_percentage'] = round(sorted_colab['Collaboration_International collaboration_percentage'] * 100, 0)

            # 평균을 구합니다.
            average = np.mean(sorted_colab['Collaboration_International collaboration_percentage'])

            # Plotly bar 차트를 그립니다.
            fig7 = px.bar(sorted_colab, x=sorted_colab.성명_ID, y='Collaboration_International collaboration_percentage', 
                        labels={'Collaboration_International collaboration_percentage':'국제공동연구 비율', '성명_ID':'연번'},
                        #title=f'{selected_dept} 교원별 국제공동연구 비율(상위 50인)',
                        hover_data=['성명', '대학교명'],
                        color_discrete_sequence=['lightskyblue']
                        )

            # x축 레이블을 '성명'만 표시하도록 업데이트합니다.
            fig7.update_xaxes(tickvals=sorted_colab.성명_ID, ticktext=sorted_colab.연번, tickangle=0)
            fig7.update_layout(height=700)

            # horizontal dash line을 그립니다.
            fig7.add_shape(type="line", x0=0, x1=1, y0=average, y1=average, xref='paper', yref='y',
                        line=dict(color="orange", width=1, dash="dash"))

            # 평균 수치를 text로 추가합니다.
            fig7.add_annotation(x=len(sorted_colab)/2, y=average, text=f"평균: {average:.2f}", showarrow=True, arrowhead=2, arrowcolor="orange", font=dict(size=14, color="darkorange"), align="center")

            # 평균 범례를 추가합니다.
            fig7.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color="orange", width=1, dash="dash"), name="평균"))

            # Streamlit에서 차트를 보여줍니다.
            st.plotly_chart(fig7, use_container_width=True)

            # 새로운 DataFrame을 다시 만듭니다.
            display_df7 = sorted_colab[['연번', '성명', '대학교명', 'Collaboration_International collaboration_percentage']].copy()
            display_df7.columns = ['연번', '성명', '대학교', "비율(%)"]

            st.dataframe(display_df7, use_container_width=True, hide_index=True)
            
            

            # '대학교명'과 'Year'별로 논문 수를 계산합니다.
            comparison_outputs = univ_comparison.groupby(['대학교명', 'Year'])['eid'].count().unstack().reset_index().fillna(0)

            # '총 논문수' 칼럼을 추가합니다.
            comparison_outputs['계'] = comparison_outputs.iloc[:, 1:].sum(axis=1)
            comparison_outputs = comparison_outputs.set_index('대학교명')
            comparison_outputs = comparison_outputs.reindex(['연세대학교'] + [index for index in comparison_outputs.index if index != '연세대학교']).reset_index()

            # 칼럼 순서를 변경합니다.
            comparison_outputs = comparison_outputs[['대학교명', '계', '2019', '2020', '2021', '2022', '2023']]

            st.divider()

            st.subheader(f'III. 국내 주요 대학 {selected_dept} 관련학과 논문성과 비교')
            st.subheader('(2019-2023)')
            st.write('')
            st.markdown('1. 개요')

            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.write('')
            st.markdown('2. [양적지표] 논문 수')
            st.markdown('&#160;&#160;&#160;가. 연도별 논문 수 및 교원 1인당 논문 수')

            colors = {
                '연세대학교': 'royalblue',
                '고려대학교': 'red',
                '서울대학교': 'darkorange',
                '성균관대학교': 'limegreen',
                '한양대학교': 'aqua',
                'KAIST': 'slategrey',
                'POSTECH': 'darkviolet',
                '서강대학교': 'darkgreen',
                '이화여자대학교': 'fuchsia',
                '부산대학교': 'darkorchid',
                '전북대학교':'olive',
                '중앙대학교': 'midnightblue',
                '경상대': 'goldenrod',
                '전남대학교': 'limegreen',
                '경희대학교': 'grey',
                '세종대학교': 'darkgoldenrod',
                '충남대학교': 'forestgreen',
                '공주대학교': 'pink',
                '부경대학교': 'darkseagreen',
                '홍익대학교': 'seagreen',
                '서울시립대학교': 'tan'
            }
            
            fig1 = go.Figure()

            # '대학교명'을 반복하며 각 대학교에 대한 선 그래프를 추가합니다.
            for university in comparison_outputs['대학교명']:
                fig1.add_trace(go.Scatter(
                    x=['2019', '2020', '2021', '2022', '2023'],
                    y=comparison_outputs[comparison_outputs['대학교명'] == university][['2019', '2020', '2021', '2022', '2023']].values.flatten(),
                    mode='lines+markers',
                    name=university,
                    line=dict(color=colors[university]),
                    marker=dict(symbol='circle', size=12, color='white', line=dict(color=colors[university], width=1.5))
                ))

            # 그래프의 제목과 축 레이블을 설정합니다.
            fig1.update_layout(
                title='연도별 논문 수',
                xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']),
                xaxis_title='Year',
            )
            fig1.update_layout(height=700)
            st.plotly_chart(fig1, use_container_width=True)

            st.dataframe(comparison_outputs, use_container_width=True, hide_index=True)

            # '대학교명'과 'Year'별로 교원 1인당 논문 수를 계산합니다.
            comparison_outputs_per_faculty = univ_comparison.groupby(['대학교명', 'Year'])['eid'].count().unstack().reset_index().fillna(0)
            comparison_outputs_per_faculty['교원 수'] = comparison_outputs_per_faculty.대학교명.map(selected_data.groupby('대학교명')['성명'].count())
            for year in ['2019', '2020', '2021', '2022', '2023']:
                comparison_outputs_per_faculty[year] = round(comparison_outputs_per_faculty[year] / comparison_outputs_per_faculty['교원 수'], 1)

            # '총 교원 1인당 논문 수' 칼럼을 추가합니다.
            comparison_outputs_per_faculty['계'] = comparison_outputs_per_faculty['대학교명'].map(comparison.groupby('대학교명')['교원 1인당 연간 논문수'].mean())

            # 칼럼 순서를 변경합니다.
            comparison_outputs_per_faculty = comparison_outputs_per_faculty[['대학교명', '계', '2019', '2020', '2021', '2022', '2023']]
            comparison_outputs_per_faculty = comparison_outputs_per_faculty.set_index('대학교명')
            comparison_outputs_per_faculty = comparison_outputs_per_faculty.reindex(['연세대학교'] + [index for index in comparison_outputs_per_faculty.index if index != '연세대학교']).reset_index()

            fig2 = go.Figure()
            # '대학교명'을 반복하며 각 대학교에 대한 선 그래프를 추가합니다.
            for university in comparison_outputs_per_faculty['대학교명']:
                fig2.add_trace(go.Scatter(
                    x=['2019', '2020', '2021', '2022', '2023'],
                    y=comparison_outputs_per_faculty[comparison_outputs_per_faculty['대학교명'] == university][['2019', '2020', '2021', '2022', '2023']].values.flatten(),
                    mode='lines+markers',
                    name=university,
                    line=dict(color=colors[university]),
                    marker=dict(symbol='circle', size=12, color='white', line=dict(color=colors[university], width=1.5))
                ))

            # 그래프의 제목과 축 레이블을 설정합니다.
            fig2.update_layout(
                title='연도별 교원 1인당 논문 수',
                xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']),
                xaxis_title='Year',
            )

            fig2.update_layout(height=700)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(comparison_outputs_per_faculty, use_container_width=True, hide_index=True)

            st.markdown('3. [질적지표] 피인용수를 기반으로 한 지표')
            st.markdown('&#160;&#160;&#160;가. 논문 1편당 피인용수')      

            # '대학교명'과 'Year'별로 논문 1편당 피인용수를 계산합니다.
            comparison_citations_per_paper = round(univ_comparison.groupby(['대학교명', 'Year'])['total_citations'].sum().unstack().reset_index().set_index('대학교명') / univ_comparison.groupby(['대학교명', 'Year'])['eid'].count().unstack().reset_index().set_index('대학교명'),1)
            comparison_citations_per_paper['계'] = comparison_citations_per_paper.index.map(univ_comparison.groupby('대학교명')['논문 1편당 피인용수'].mean())
            comparison_citations_per_paper = comparison_citations_per_paper[['계', '2019', '2020', '2021', '2022', '2023']]
            comparison_citations_per_paper = comparison_citations_per_paper.reindex(['연세대학교'] + [index for index in comparison_citations_per_paper.index if index != '연세대학교']).reset_index()

            fig3 = go.Figure()
            # '대학교명'을 반복하며 각 대학교에 대한 선 그래프를 추가합니다.
            for university in comparison_citations_per_paper['대학교명']:
                fig3.add_trace(go.Scatter(
                    x=['2019', '2020', '2021', '2022', '2023'],
                    y=comparison_citations_per_paper[comparison_citations_per_paper['대학교명'] == university][['2019', '2020', '2021', '2022', '2023']].values.flatten(),
                    mode='lines+markers',
                    name=university,
                    line=dict(color=colors[university]),
                    marker=dict(symbol='circle', size=12, color='white', line=dict(color=colors[university], width=1.5))
                ))

            # 그래프의 제목과 축 레이블을 설정합니다.
            fig3.update_layout(
                title='연도별 논문 1편당 피인용수',
                xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']),
                xaxis_title='Year',
            )

            fig3.update_layout(height=700)
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(comparison_citations_per_paper, use_container_width=True, hide_index=True)

            st.markdown('&#160;&#160;&#160;나. FWCI')

            # '대학교명'과 'Year'별로 논문 1편당 피인용수를 계산합니다.
            comparison_fwci = round(univ_comparison.groupby(['대학교명', 'Year'])['FieldWeightedCitationImpact'].mean().unstack().reset_index().set_index('대학교명'),2)
            comparison_fwci['계'] = comparison_fwci.index.map(univ_comparison.groupby('대학교명')['평균 FWCI'].mean())
            comparison_fwci = comparison_fwci[['계', '2019', '2020', '2021', '2022', '2023']]
            comparison_fwci = comparison_fwci.reindex(['연세대학교'] + [index for index in comparison_fwci.index if index != '연세대학교']).reset_index()

            fig4 = go.Figure()
            # '대학교명'을 반복하며 각 대학교에 대한 선 그래프를 추가합니다.
            for university in comparison_fwci['대학교명']:
                fig4.add_trace(go.Scatter(
                    x=['2019', '2020', '2021', '2022', '2023'],
                    y=comparison_fwci[comparison_fwci['대학교명'] == university][['2019', '2020', '2021', '2022', '2023']].values.flatten(),
                    mode='lines+markers',
                    name=university,
                    line=dict(color=colors[university]),
                    marker=dict(symbol='circle', size=12, color='white', line=dict(color=colors[university], width=1.5))
                ))

            # 그래프의 제목과 축 레이블을 설정합니다.
            fig4.update_layout(
                title='연도별 FWCI',
                xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']),
                xaxis_title='Year',
            )

            fig4.update_layout(height=700)
            st.plotly_chart(fig4, use_container_width=True)

            st.dataframe(comparison_fwci, use_container_width=True, hide_index=True)
            
            st.markdown('&#160;&#160;&#160;다. FWCI 상위 10% 논문 비율')

            # '대학교명'과 'Year'별로 논문 1편당 피인용수를 계산합니다.
            comparison_fwci_top_10 = round(100*(univ_comparison.groupby(['대학교명', 'Year'])['OutputsInTopCitationPercentiles_10'].sum().unstack() / univ_comparison.groupby(['대학교명', 'Year'])['eid'].count().unstack()),1)
            comparison_fwci_top_10['계'] = comparison_fwci_top_10.index.map(univ_comparison.groupby('대학교명')['FWCI 상위 10% 논문비율(%)'].mean())
            comparison_fwci_top_10 = comparison_fwci_top_10[['계', '2019', '2020', '2021', '2022', '2023']]
            comparison_fwci_top_10 = comparison_fwci_top_10.reindex(['연세대학교'] + [index for index in comparison_fwci_top_10.index if index != '연세대학교']).reset_index()

            fig5 = go.Figure()
            # '대학교명'을 반복하며 각 대학교에 대한 선 그래프를 추가합니다.
            for university in comparison_fwci_top_10['대학교명']:
                fig5.add_trace(go.Scatter(
                    x=['2019', '2020', '2021', '2022', '2023'],
                    y=comparison_fwci_top_10[comparison_fwci_top_10['대학교명'] == university][['2019', '2020', '2021', '2022', '2023']].values.flatten(),
                    mode='lines+markers',
                    name=university,
                    line=dict(color=colors[university]),
                    marker=dict(symbol='circle', size=12, color='white', line=dict(color=colors[university], width=1.5))
                ))

            # 그래프의 제목과 축 레이블을 설정합니다.
            fig5.update_layout(
                title='연도별 FWCI 상위 10% 논문 비율(%)',
                xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']),
                xaxis_title='Year',
            )

            fig5.update_layout(height=700)
            st.plotly_chart(fig5, use_container_width=True)

            st.dataframe(comparison_fwci_top_10, use_container_width=True, hide_index=True)

            st.markdown('&#160;&#160;&#160;라. 상위 10% 저널에 게재된 논문 비율 (SNIP기준)')

            # '대학교명'과 'Year'별로 논문 1편당 피인용수를 계산합니다.
            comparison_snip_top_10 = round(100*(univ_comparison.groupby(['대학교명', 'Year'])['PublicationsInTopJournalPercentiles_10'].sum().unstack() / univ_comparison.groupby(['대학교명', 'Year'])['eid'].count().unstack()),1)
            comparison_snip_top_10['계'] = comparison_snip_top_10.index.map(univ_comparison.groupby('대학교명')['SNIP기준 상위 10% 저널 게재비율(%)'].mean())
            comparison_snip_top_10 = comparison_snip_top_10[['계', '2019', '2020', '2021', '2022', '2023']]
            comparison_snip_top_10 = comparison_snip_top_10.reindex(['연세대학교'] + [index for index in comparison_snip_top_10.index if index != '연세대학교']).reset_index()

            fig6 = go.Figure()
            # '대학교명'을 반복하며 각 대학교에 대한 선 그래프를 추가합니다.
            for university in comparison_snip_top_10['대학교명']:
                fig6.add_trace(go.Scatter(
                    x=['2019', '2020', '2021', '2022', '2023'],
                    y=comparison_snip_top_10[comparison_snip_top_10['대학교명'] == university][['2019', '2020', '2021', '2022', '2023']].values.flatten(),
                    mode='lines+markers',
                    name=university,
                    line=dict(color=colors[university]),
                    marker=dict(symbol='circle', size=12, color='white', line=dict(color=colors[university], width=1.5))
                ))

            # 그래프의 제목과 축 레이블을 설정합니다.
            fig6.update_layout(
                title='연도별 SNIP기준 상위 10% 저널 게재비율(%)',
                xaxis=dict(tickvals=['2019', '2020', '2021', '2022', '2023']),
                xaxis_title='Year',
            )

            fig6.update_layout(height=700)
            st.plotly_chart(fig6, use_container_width=True)

            st.dataframe(comparison_snip_top_10, use_container_width=True, hide_index=True)

            st.markdown('4. 공동연구')
            st.markdown('&#160;&#160;&#160;가. 공동연구 유형별 비율')
            # Assuming that the result of the groupby operation is stored in the variable 'collaboration'
            collaboration = univ_comparison.groupby('대학교명')[['InstitutionalCollaboration', 'InternationalCollaboration', 'NationalCollaboration', 'SingleAuthorship']].agg('mean')

            # Define colors for each type of collaboration
            colors = ['palegoldenrod', 'lightskyblue', 'aliceblue', 'dimgray']

            # Define the order of universities
            university_order = ['연세대학교'] + [univ for univ in collaboration.index if univ != '연세대학교']

            # Create traces for each type of collaboration
            trace1 = go.Bar(x=university_order, y=collaboration.loc[university_order, 'InternationalCollaboration'], name='International Collaboration', marker_color=colors[0], width=0.3, text=collaboration.loc[university_order, 'InternationalCollaboration'].apply(lambda x: f'{x*100:.1f}%'), textposition='auto')
            trace2 = go.Bar(x=university_order, y=collaboration.loc[university_order, 'NationalCollaboration'], name='National Collaboration', marker_color=colors[1], width=0.3, text=collaboration.loc[university_order, 'NationalCollaboration'].apply(lambda x: f'{x*100:.1f}%'), textposition='auto')
            trace3 = go.Bar(x=university_order, y=collaboration.loc[university_order, 'InstitutionalCollaboration'], name='Institutional Collaboration', marker_color=colors[2], width=0.3, text=collaboration.loc[university_order, 'InstitutionalCollaboration'].apply(lambda x: f'{x*100:.1f}%'), textposition='auto')
            trace4 = go.Bar(x=university_order, y=collaboration.loc[university_order, 'SingleAuthorship'], name='Single Authorship', marker_color=colors[3], width=0.3, text=collaboration.loc[university_order, 'SingleAuthorship'].apply(lambda x: f'{x*100:.1f}%'), textposition='auto')

            # Create the figure and add the traces
            fig7 = go.Figure(data=[trace1, trace2, trace3, trace4])

            # Update the layout to stack the bars
            fig7.update_layout(barmode='stack', height=600)
            fig7.update_yaxes(tickformat=".0%")

            # Display the figure in Streamlit
            st.plotly_chart(fig7, use_container_width=True)

            
            st.markdown('&#160;&#160;&#160;나. 공동연구 유형별 FWCI')

            # 공동연구 유형 리스트 생성
            collaboration_types = ['InternationalCollaboration', 'NationalCollaboration', 'InstitutionalCollaboration', 'SingleAuthorship']

            # 결과를 저장할 빈 DataFrame 초기화
            results = pd.DataFrame()

            # 각 공동연구 유형에 대해 반복
            for collaboration_type in collaboration_types:
                # 공동연구 유형이 1인 행으로 DataFrame 필터링
                filtered_df = univ_comparison[univ_comparison[collaboration_type] == 1]
                
                # 각 대학교별 'FWCI'의 평균 계산
                mean_FWCI = filtered_df.groupby('대학교명')['FieldWeightedCitationImpact'].mean()
                
                # 결과를 결과 DataFrame에 추가
                results[collaboration_type] = mean_FWCI

            results = round(results,2)
            results = results.fillna(0)

            # 'results' DataFrame을 재설정하여 각 대학교별로 공동연구 유형의 'FWCI' 평균을 얻습니다.
            df_melted = results.reset_index().melt(id_vars='대학교명', var_name='공동연구 유형', value_name='평균 FWCI')

            # 색상 매핑 정의
            colors = {
                'InternationalCollaboration': 'palegoldenrod',
                'NationalCollaboration': 'lightskyblue',
                'InstitutionalCollaboration': 'aliceblue',
                'SingleAuthorship': 'dimgray'
            }

            # 대학교명 순서 지정
            university_order = ['연세대학교'] + [univ for univ in df_melted['대학교명'].unique() if univ != '연세대학교']

            fig8 = px.bar(df_melted, x='대학교명', y='평균 FWCI', color='공동연구 유형', barmode='group', text='평균 FWCI', color_discrete_map=colors,
                        category_orders={'대학교명': university_order})
            fig8.update_traces(texttemplate='%{text:.2f}', textposition='outside', width=0.1)
            fig8.update_layout(xaxis_title=None, height=600)


            # 플롯을 보여줍니다.
            st.plotly_chart(fig8, use_container_width=True)

            percentile = round(univ_comparison.groupby('대학교명')[['InstitutionalCollaboration', 'InternationalCollaboration', 'NationalCollaboration', 'SingleAuthorship']].agg('mean'),3)
            percentile['계'] = 1
            percentile = percentile[['계', 'InternationalCollaboration', 'NationalCollaboration', 'InstitutionalCollaboration', 'SingleAuthorship']]*100
            percentile = percentile.rename(columns={'InternationalCollaboration':'국제 공동연구 비율(%)', 'NationalCollaboration':'국내 공동연구 비율(%)', 'InstitutionalCollaboration':'기관내 공동연구 비율(%)', 'SingleAuthorship':'단독연구 비율(%)'})

            results['평균 FWCI'] = results.index.map(round(univ_comparison.groupby('대학교명')['FieldWeightedCitationImpact'].mean(),2))
            results = results.rename(columns={'InternationalCollaboration':'국제 공동연구 FWCI', 'NationalCollaboration':'국내 공동연구 FWCI', 'InstitutionalCollaboration':'기관내 공동연구 FWCI', 'SingleAuthorship':'단독연구 FWCI'})
            results = results[['평균 FWCI', '국제 공동연구 FWCI', '국내 공동연구 FWCI', '기관내 공동연구 FWCI', '단독연구 FWCI']]
            df_collaboration = pd.concat([percentile, results], axis=1)
            df_collaboration = df_collaboration.reindex(['연세대학교'] + [index for index in df_collaboration.index if index != '연세대학교']).reset_index()

            st.dataframe(df_collaboration, use_container_width=True, hide_index=True)

    finally:
        connection.close()
   
        
if __name__ == "__main__":
    main()
