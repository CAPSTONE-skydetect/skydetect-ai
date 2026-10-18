import pytest

from research.ard100_benchmark import parse_voc


def xml(objects):
    return ("<annotation><size><width>1920</width><height>1080</height></size>"
            + objects + "</annotation>").encode()


BOX = ("<object><name>Drone</name><bndbox><xmin>471</xmin><ymin>310</ymin>"
       "<xmax>507</xmax><ymax>334</ymax></bndbox></object>")


def test_preserve_annotation_coordinates_without_one_pixel_guess():
    assert parse_voc(xml(BOX)) == (1920, 1080, [471., 310., 507., 334.])


def test_empty_annotation_means_no_observation_not_a_fabricated_box():
    assert parse_voc(xml("")) == (1920, 1080, None)


def test_ambiguous_multiple_objects_are_not_silently_merged():
    with pytest.raises(ValueError, match="exactly one"):
        parse_voc(xml(BOX + BOX))


def test_label_and_bounds_are_checked():
    with pytest.raises(ValueError, match="Drone"):
        parse_voc(xml(BOX.replace("Drone", "Bird")))
    with pytest.raises(ValueError, match="outside"):
        parse_voc(xml(BOX.replace("<xmin>471", "<xmin>-1")))
