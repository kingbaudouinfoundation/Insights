def apply_area_filter(df, filter_args, filter_def):

    if not filter_args or filter_args == ['__all']:
        return

    countries = []
    regions = []
    for f in filter_args:
        if "##" in f:
            f = f.split('##')
            countries.append(f[0])
            regions.append(f[1])
    if countries and regions:
        return df[
            (df["__geo_ctry"].isin(countries)) &
            (df["__geo_rgn"].isin(regions))
        ]


def apply_field_filter(df, filter_args, filter_def):

    if not filter_args or filter_args == ['__all']:
        return

    

    return df[df[filter_def["field"]].isin(filter_args)]

def apply_date_range_filter(df, filter_args, filter_def):
    if not filter_args or df is None:
        return

    return df[
        (df[filter_def["field"]].dt.year >= filter_args[0]) &
        (df[filter_def["field"]].dt.year <= filter_args[1])
    ]

def apply_range_filter(df, filter_args, filter_def):
    if not filter_args or df is None:
        return

    return df[
        (df[filter_def["field"]] >= filter_args[0]) &
        (df[filter_def["field"]] <= filter_args[1])
    ]

FILTERS = {
    "funders": {
        "label": "Funders",
        "type": "multidropdown",
        "defaults": [{"label": "Funder", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': '{} ({})'.format(i[0], i[1]),
                'value': i[0]
            } for i in df["Funding Org:Name"].value_counts().iteritems()
        ]),
        "field": "Funding Org:Name",
        "apply_filter": apply_field_filter,
    },
    "grant_programmes": {
        "label": "Grant programmes",
        "type": "multidropdown",
        "defaults": [{"label": "All grants", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': '{} ({})'.format(i[0], i[1]),
                'value': i[0]
            } for i in df["Grant Programme:Title"].value_counts().iteritems()
        ]),
        "field": "Grant Programme:Title",
        "apply_filter": apply_field_filter,
    },
    "award_dates": {
        "label": "Date awarded",
        "type": "rangeslider",
        "defaults": {"min": 2015, "max": 2018},
        "get_values": (lambda df: {
            "min": int(df["Award Date"].dt.year.min()),
            "max": int(df["Award Date"].dt.year.max()),
        }),
        "field": "Award Date",
        "apply_filter": apply_date_range_filter,
    },
    "area": {
        "label": "Region and country",
        "type": "multidropdown",
        "defaults": [{"label": "All areas", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': (
                    '{} ({})'.format(value[0], count)
                    if value[0].strip() == value[1].strip()
                    else "{} - {} ({})".format(value[0], value[1], count)
                ),
                'value': "{}##{}".format(value[0], value[1])
            } for value, count in df.fillna({"__geo_ctry": "Unknown", "__geo_rgn": "Unknown"}).groupby(["__geo_ctry", "__geo_rgn"]).size().iteritems()
        ]),
        "apply_filter": apply_area_filter,
    },
    "orgtype": {
        "label": "Organisation type",
        "type": "multidropdown",
        "defaults": [{"label": "All organisation types", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': '{} ({})'.format(i[0], i[1]),
                'value': i[0]
            } for i in df["__org_org_type"].value_counts().iteritems()
        ]),
        "field": "__org_org_type",
        "apply_filter": apply_field_filter,
    },
    "award_amount": {
        "label": "Amount awarded",
        "type": "multidropdown",
        "defaults": [{"label": "All amounts", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': '{} ({})'.format(i[0], i[1]),
                'value': i[0]
            } for i in df["Amount Awarded:Bands"].value_counts().sort_index().iteritems()
        ]),
        "field": "Amount Awarded:Bands",
        "apply_filter": apply_field_filter,
    },
    "org_size": {
        "label": "Organisation size",
        "type": "multidropdown",
        "defaults": [{"label": "All sizes", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': '{} ({})'.format(i[0], i[1]),
                'value': i[0]
            } for i in df["__org_latest_income_bands"].value_counts().sort_index().iteritems()
        ] if df["__org_latest_income_bands"].value_counts().sum() else []),
        "field": "__org_latest_income_bands",
        "apply_filter": apply_field_filter,
    },
    "org_age": {
        "label": "Organisation age",
        "type": "multidropdown",
        "defaults": [{"label": "All ages", "value": "__all"}],
        "get_values": (lambda df: [
            {
                'label': '{} ({})'.format(i[0], i[1]),
                'value': i[0]
            } for i in df["__org_age_bands"].value_counts().sort_index().iteritems()
        ] if df["__org_age_bands"].value_counts().sum() else []),
        "field": "__org_age_bands",
        "apply_filter": apply_field_filter,
    },
}