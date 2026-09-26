def build_box_with_score_map_label_generator(config: dict, plugin_config: dict = None):
    common_config = config['common']
    plugin_config = plugin_config or {}
    from . import BoxWithScoreMapLabelGenerator, box_with_score_map_label_collator
    return (BoxWithScoreMapLabelGenerator(
                common_config['response_map_size'],
                common_config['search_region_size'],
                plugin_config.get('positive_assignment', 'box'),
                plugin_config.get('center_positive_radius', 1)),
            box_with_score_map_label_collator)
