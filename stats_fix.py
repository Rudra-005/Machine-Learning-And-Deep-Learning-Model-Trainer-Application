        st.subheader("📊 Data Statistics")
        try:
            X_df = pd.DataFrame(st.session_state.X_train)
            X_numeric = X_df.apply(pd.to_numeric, errors='coerce')
            stats_df = pd.DataFrame({
                'Mean': X_numeric.mean(),
                'Std': X_numeric.std(),
                'Min': X_numeric.min(),
                'Max': X_numeric.max()
            })
            st.dataframe(stats_df, use_container_width=True)
        except Exception as e:
            st.info("Statistics not available for this data type")
