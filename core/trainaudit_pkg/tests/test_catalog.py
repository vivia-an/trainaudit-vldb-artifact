from trainaudit.catalog import (
    CATALOG_BY_ID,
    FROZEN_CATALOG_SHA256,
    catalog_templates,
    get_catalog_template,
)


def test_frozen_catalog_index_is_complete_and_unique():
    templates = list(catalog_templates())
    assert len(templates) == 35
    assert len(CATALOG_BY_ID) == 35
    assert [template.template_id for template in templates] == [
        f"T{i:02d}" for i in range(1, 36)
    ]
    assert len({template.name for template in templates}) == 35
    assert len(FROZEN_CATALOG_SHA256) == 64


def test_catalog_lookup_uses_canonical_id_and_name():
    template = get_catalog_template("T01")
    assert template.name == "cross-rank-replica-equality"
    assert template.relation_operator == "equality_across_ranks"
