databricks_config = {
    "catalog": "hive_metastore",
    "schema": "default",
    "host": "https://adb-320373841791891.11.azuredatabricks.net/",
    "table_name": "processed_reservations",
    "experiment_name_basic": "/Shared/basic_reservations_experiment_willem",
    "experiment_name_fe": "/Shared/fe_reservations_experiment_willem",
    "num_features": [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "type_of_meal_plan_meal_plan_1",
        "type_of_meal_plan_meal_plan_2",
        "type_of_meal_plan_meal_plan_3",
        "type_of_meal_plan_not_selected",
        "room_type_reserved_room_type_1",
        "room_type_reserved_room_type_2",
        "room_type_reserved_room_type_3",
        "room_type_reserved_room_type_4",
        "room_type_reserved_room_type_5",
        "room_type_reserved_room_type_6",
        "room_type_reserved_room_type_7",
        "market_segment_type_aviation",
        "market_segment_type_complementary",
        "market_segment_type_corporate",
        "market_segment_type_offline",
        "market_segment_type_online",
    ],
    "num_features_fe": ["lead_time_cat"],
    "target": "booking_status",
}
