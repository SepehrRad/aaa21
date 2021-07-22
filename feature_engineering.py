def add_weekday(df):
    """
    This function adds weekdays (Sunday, Monday,..) to the data frame.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to which weekday names should be added.
    :return
        pandas.DataFrame: The data frame with additional weekday column
    """
    # Creating Weekday columns
    df["Pickup Day"] = df["Trip Start Timestamp"].dt.day_name()
    df["Drop-off Day"] = df["Trip End Timestamp"].dt.day_name()
    return df


def add_time_interval(df):
    """
    This function adds time zones (morning, evening, etc.) based on the hour of the day.
    ----------------------------------------------
    :param
        df(pd.DataFrame): DataFrame to which time zone information should be added.
    """
    # Pickup time interval
    df.loc[
        df["Trip Start Hour"].between(5, 11, inclusive="left"), "Pickup Time_Interval"
    ] = "morning"
    df.loc[
        df["Trip Start Hour"].between(11, 17, inclusive="left"), "Pickup Time_Interval"
    ] = "midday"
    df.loc[
        df["Trip Start Hour"].between(17, 23, inclusive="left"), "Pickup Time_Interval"
    ] = "evening"
    df.loc[
        ((df["Trip Start Hour"] >= 23) | (df["Trip Start Hour"] < 5)),
        "Pickup Time_Interval",
    ] = "night"

    # Drop-off time interval
    df.loc[
        df["Trip End Hour"].between(5, 11, inclusive="left"), "Drop-off Time_Interval"
    ] = "morning"
    df.loc[
        df["Trip End Hour"].between(11, 17, inclusive="left"), "Drop-off Time_Interval"
    ] = "midday"
    df.loc[
        df["Trip End Hour"].between(17, 23, inclusive="left"), "Drop-off Time_Interval"
    ] = "evening"
    df.loc[
        ((df["Trip End Hour"] >= 23) | (df["Trip End Hour"] < 5)),
        "Drop-off Time_Interval",
    ] = "night"
