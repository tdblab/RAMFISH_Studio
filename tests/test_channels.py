"""Which round and gene every channel of a multi-channel file is."""

import pytest

from ramfish.channels import (HOW_DYE_NAME, HOW_ORDER, HOW_WAVELENGTH, ChannelMeta, assign_channels,
                              dye_key, parse_filename)


def stains(name):
    p = parse_filename(name)
    return p.round, p.prefix, [(s.gene, s.dye or s.amplifier) for s in p.stains]


@pytest.mark.parametrize("name, expected", [
    ("round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514__1.oir",
     (1, "", [("dpp", "alexa546"), ("optix", "alexa647"), ("omb", "alexa488"), ("hh", "alexa514")])),
    ("round3_WntA_alexa546__notch_alex647__mad6_alexa488__dll_alex514_.oir",
     (3, "", [("WntA", "alexa546"), ("notch", "alexa647"), ("mad6", "alexa488"), ("dll", "alexa514")])),
    ("fish1_round3_tph2 _elavl3_chatb.oir", (3, "fish1", [("tph2", None), ("elavl3", None), ("chatb", None)])),
    ("R2_spalt(b1)_optix(B2)_omb(b3)_DAPI_40x_oil.czi",
     (2, "", [("spalt", "b1"), ("optix", "b2"), ("omb", "b3"), ("DAPI", "dapi")])),
    ("gad2_AF546_nrp1a_488_wt.tif", (None, "", [("gad2", "alexa546"), ("nrp1a", "alexa488")])),
    ("round2_gfp_alexa488.tif", (2, "", [("gfp", "alexa488")])),
    ("round12_hsp67balexa546_rho3alexa647_ombalexa488.oir",
     (12, "", [("hsp67b", "alexa546"), ("rho3", "alexa647"), ("omb", "alexa488")])),
])
def test_parse_filename(name, expected):
    assert stains(name) == expected


def test_dye_names():
    assert dye_key("Alexa Fluor 546") == "alexa546"
    assert dye_key("AF647") == dye_key("A647") == dye_key("647") == dye_key("alex647") == "alexa647"
    assert dye_key("Hoechst 33342") == "hoechst" and dye_key("dpp") is None


# The Olympus files store 488, 514, 546, 647 in that order, whatever order the file name uses.
OIR_META = [ChannelMeta(0, "CH1", "alexa488", 500.0), ChannelMeta(1, "CH2", "alexa514", 530.0),
            ChannelMeta(2, "CH3", "alexa546", 570.0), ChannelMeta(3, "CH4", "alexa647", 650.0)]
ROUND1 = "round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514_.oir"


def test_dye_names_saved_in_the_file_win():
    out, warns = assign_channels(parse_filename(ROUND1), OIR_META)
    assert [a.name for a in out] == ["omb", "hh", "dpp", "optix"]
    assert {a.how for a in out} == {HOW_DYE_NAME} and not warns


@pytest.mark.parametrize("name", [ROUND1, "round1_hh_alexa514__dpp_alexa546__omb_alexa488__optix_alexa647.oir"])
def test_wavelengths_alone_give_the_same_answer(name):
    """514 nm is closer to the 500 nm channel than to its own (530 nm): only the best overall match is right."""
    meta = [ChannelMeta(c.index, c.name, None, c.excitation_nm) for c in OIR_META]
    out, _ = assign_channels(parse_filename(name), meta)
    assert [a.name for a in out] == ["omb", "hh", "dpp", "optix"]
    assert {a.how for a in out} == {HOW_WAVELENGTH}


def test_unnamed_405_channel_is_dapi():
    meta = [ChannelMeta(0, "", None, 405.0), ChannelMeta(1, "", None, 488.0), ChannelMeta(2, "", None, 561.0)]
    out, _ = assign_channels(parse_filename("round1_wg_alexa546_sal_alexa488.czi"), meta)
    assert [a.name for a in out] == ["DAPI", "sal", "wg"]


def test_amplifiers_use_the_table():
    meta = [ChannelMeta(0, "", None, 488.0), ChannelMeta(1, "", None, 561.0), ChannelMeta(2, "", None, 640.0)]
    out, _ = assign_channels(parse_filename("round1_spalt(b1)_optix(b2)_omb(b3).oir"), meta)
    assert [a.name for a in out] == ["omb", "spalt", "optix"]
    out, _ = assign_channels(parse_filename("round1_spalt(b1)_optix(b2)_omb(b3).oir"), meta,
                             {"b1": "alexa488", "b2": "alexa546", "b3": "alexa647"})
    assert [a.name for a in out] == ["spalt", "optix", "omb"]


def test_no_information_falls_back_to_name_order_and_says_so():
    out, warns = assign_channels(parse_filename("fish1_round1_gad2_neuroD_nrp1a.oir"), [ChannelMeta(i) for i in range(4)])
    assert [a.name for a in out] == ["gad2", "neuroD", "nrp1a", "C4"]
    assert out[0].how == HOW_ORDER


def test_too_many_genes_is_reported():
    _, warns = assign_channels(parse_filename("round1_a_b_c.tif"), [ChannelMeta(0), ChannelMeta(1)])
    assert any("no channel left" in w for w in warns)


# ---------------------------------------------------------------- extra _00001 files of large .oir stacks
def test_oir_parts_are_found_next_to_their_oir():
    from ramfish.channels import oir_part_stem, oir_parts_for
    oir = "Round_2/round2_DAPI_oxt(alexa546)_kctd12.2(alexa647)_omp(alexa488).oir"
    names = [oir,
             "Round_2/round2_DAPI_oxt(alexa546)_kctd12.2(alexa647)_omp(alexa488)_00002",
             "Round_2/round2_DAPI_oxt(alexa546)_kctd12.2(alexa647)_omp(alexa488)_00001",
             "Round_3/round2_DAPI_oxt(alexa546)_kctd12.2(alexa647)_omp(alexa488)_00001",   # other folder
             "Round_2/other_scan_00001", "Round_2/notes.txt"]
    assert oir_parts_for(oir, names) == [names[2], names[1]]
    assert oir_part_stem("a/b/scan_1_00001") == "scan_1"
    assert oir_part_stem("scan_0001") is None and oir_part_stem("scan.oir") is None
    assert oir_parts_for("scan.tif", ["scan_00001"]) == []


def test_parts_are_not_taken_as_images():
    from ramfish import dataset
    names = ["Round_2/round2_DAPI_oxt(alexa546)_kctd12.2(alexa647)_omp(alexa488).oir",
             "Round_2/round2_DAPI_oxt(alexa546)_kctd12.2(alexa647)_omp(alexa488)_00001"]
    assert [e.name for e in dataset.scan(names)] == names[:1]


def test_planned_z_from_oir_settings(tmp_path):
    from ramfish.channels import oir_planned_z
    xml = (b"\x00" * 1000 + b'<commonimage:axis xsi:type="commonparam:ZAxisParam"> <commonparam:axis>ZSTACK</commonparam:axis>'
           b" <commonparam:startPosition>2275.0</commonparam:startPosition> <commonparam:endPosition>2345.18"
           b"</commonparam:endPosition> <commonparam:step>3.19</commonparam:step> <commonparam:maxSize>23"
           b"</commonparam:maxSize> </commonimage:axis>")
    p = tmp_path / "a.oir"
    p.write_bytes(xml)
    assert oir_planned_z(p) == (23, 3.19)
    q = tmp_path / "b.oir"
    q.write_bytes(b"\x00" * 100)
    assert oir_planned_z(q) is None


def test_zip_source_copies_members(tmp_path):
    import io, zipfile
    from ramfish.dataset import ZipSource
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Round_1/scan_00001", b"abc" * 1000)
    ZipSource(buf).copy_to("Round_1/scan_00001", tmp_path / "scan_00001")
    assert (tmp_path / "scan_00001").read_bytes() == b"abc" * 1000


def test_planned_z_ignores_switched_off_z_settings(tmp_path):
    """Every .oir stores the microscope's Z settings; a single-plane image has them with enable="false"."""
    from ramfish.channels import oir_planned_z
    off = (b'<commonparam:axis xsi:type="commonparam:ZAxisParam" enable="false" paramEnable="true"> '
           b"<commonparam:axis>ZSTACK</commonparam:axis> <commonparam:startPosition>1.0</commonparam:startPosition> "
           b"<commonparam:endPosition>13.0</commonparam:endPosition> <commonparam:step>1.0</commonparam:step> "
           b"<commonparam:maxSize>13</commonparam:maxSize> </commonparam:axis>")
    p = tmp_path / "wing.oir"
    p.write_bytes(b"\x00" * 100 + off)
    assert oir_planned_z(p) is None
    on = off.replace(b'enable="false"', b'enable="true"')
    q = tmp_path / "stack.oir"
    q.write_bytes(b"\x00" * 100 + on)
    assert oir_planned_z(q) == (13, 1.0)
