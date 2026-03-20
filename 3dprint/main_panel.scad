//
// Sloped screen/front panel with full-width bottom glue lip
// and upper angled lip
//
// - centered screen cutout: 158 x 88 mm
// - screen keepout behind the panel, so other parts are not too close
// - Raspberry Pi AI Camera cutout
// - PIR cutout: 23 mm
// - 2x Same Sky pushbutton cutouts: 11.6 mm
// - full-width bottom glue lip
// - upper angled lip with 1 mm inset
// - side tray walls as wedges so they close into the upper lip
//
// Export:
//   part = "panel";   // STL
//   part = "preview"; // with dummy geometry
//

part = "panel";   // "panel" | "preview"

// =========================
// MAIN SIZE
// =========================

// visible main panel
panel_w = 200.0;
panel_h = 185.0;

// bottom glue lip: SAME WIDTH as panel
glue_lip_w = panel_w;
glue_lip_h = 18.0;

plate_t = 2.0;
outer_corner_r = 2.0;

// tray walls / "Wanne"
rim_inset = 1.0;
rim_t     = 1.2;
rim_depth = 20.0;

// upper angled lip
top_lip_t         = rim_t;
top_lip_depth     = 16.0;
top_lip_angle_deg = 5.0;   // if you meant relative to the existing 60°/30° section, try 25.0

// =========================
// SCREEN + KEEPOUT
// =========================

// visible cutout
screen_w = 158.0;
screen_h = 88.0;
screen_r = 2.0;
screen_cx = 0.0;
screen_cy = -6.0;

// larger real module behind the opening
screen_keepout_w = 172.0;
screen_keepout_h = 104.0;

// clearances from keepout to neighboring cutouts
upper_clear = 14.0;
lower_clear = 14.0;

// =========================
// CAMERA / PIR / BUTTONS
// =========================

// Raspberry Pi AI Camera
// default: round lens hole only
use_camera_rect = false;
camera_hole_d = 12.0;
camera_rect_w = 14.0;
camera_rect_h = 14.0;
camera_rect_r = 1.0;

// PIR
pir_hole_d = 23.0;

// Buttons
button_hole_d = 11.6;
button_spacing = 50.0;  // center-to-center

// layout logic
upper_row_cy = screen_cy + screen_keepout_h/2 + upper_clear + max(camera_hole_d/2, pir_hole_d/2);
button_cy    = screen_cy - screen_keepout_h/2 - lower_clear - button_hole_d/2;

// positions
camera_cx = 0.0;
camera_cy = upper_row_cy;

pir_cx = 62.0;
pir_cy = upper_row_cy;

button1_cx = -button_spacing/2;
button2_cx =  button_spacing/2;

// =========================
// PREVIEW DUMMIES
// =========================

dummy_screen_t = 4.0;
dummy_camera_d = 11.9;
dummy_pir_d    = 23.0;
dummy_button_d = 11.6;

show_keepout_preview = true;

// =========================
// DERIVED VALUES
// =========================

main_bottom_y = -panel_h/2;
main_top_y    =  panel_h/2;

y_bottom_in   = main_bottom_y + rim_inset;
y_top_in      = main_top_y    - rim_inset;

top_lip_rise = top_lip_depth * tan(top_lip_angle_deg);

y_lip_front_center = y_top_in - top_lip_t/2;
y_lip_back_center  = y_lip_front_center + top_lip_rise;

y_lip_back_high = y_lip_back_center + top_lip_t/2;

// basic sanity
assert(glue_lip_w == panel_w, "Bottom glue lip must have the same width as the panel.");
assert(camera_cy + max(camera_hole_d/2, camera_rect_h/2) < main_top_y - 4, "Camera too close to top edge.");
assert(pir_cy + pir_hole_d/2 < main_top_y - 4, "PIR too close to top edge.");
assert(button_cy - button_hole_d/2 > main_bottom_y + 4, "Buttons too close to bottom edge.");

// =========================
// 2D HELPERS
// =========================

module rounded_rect_2d(w, h, r) {
    rr = min(r, min(w, h) / 2);
    if (rr <= 0) {
        square([w, h], center = true);
    } else {
        hull() {
            for (x = [-w/2 + rr, w/2 - rr])
                for (y = [-h/2 + rr, h/2 - rr])
                    translate([x, y]) circle(r = rr, $fn = 32);
        }
    }
}

module panel_outline_2d() {
    union() {
        // main panel
        rounded_rect_2d(panel_w, panel_h, outer_corner_r);

        // full-width bottom glue lip
        translate([0, main_bottom_y - glue_lip_h/2 + 0.01])
            square([glue_lip_w, glue_lip_h + 0.02], center = true);
    }
}

module screen_cutout_2d() {
    translate([screen_cx, screen_cy])
        rounded_rect_2d(screen_w, screen_h, screen_r);
}

module camera_cutout_2d() {
    translate([camera_cx, camera_cy]) {
        if (use_camera_rect) {
            rounded_rect_2d(camera_rect_w, camera_rect_h, camera_rect_r);
        } else {
            circle(d = camera_hole_d, $fn = 64);
        }
    }
}

module pir_cutout_2d() {
    translate([pir_cx, pir_cy])
        circle(d = pir_hole_d, $fn = 72);
}

module button_cutouts_2d() {
    translate([button1_cx, button_cy])
        circle(d = button_hole_d, $fn = 64);

    translate([button2_cx, button_cy])
        circle(d = button_hole_d, $fn = 64);
}

// =========================
// FRONT PLATE
// =========================

module front_plate() {
    linear_extrude(height = plate_t, convexity = 10)
        difference() {
            panel_outline_2d();
            screen_cutout_2d();
            camera_cutout_2d();
            pir_cutout_2d();
            button_cutouts_2d();
        }
}

// =========================
// SIDE WALLS AS WEDGES
// =========================
//
// They start at the main panel area only
// and rise toward the upper angled lip,
// so there is no side gap.
//

module side_wall_wedge(x_outer) {
    eps = 0.02;

    front_h = y_top_in - y_bottom_in;
    back_h  = y_lip_back_high - y_bottom_in;

    hull() {
        // front strip
        translate([
            x_outer + rim_t/2,
            (y_bottom_in + y_top_in)/2,
            plate_t + eps/2
        ])
            cube([rim_t, front_h, eps], center = true);

        // rear strip, taller to meet upper lip
        translate([
            x_outer + rim_t/2,
            (y_bottom_in + y_lip_back_high)/2,
            plate_t + rim_depth - eps/2
        ])
            cube([rim_t, back_h, eps], center = true);
    }
}

module left_wall() {
    side_wall_wedge(-panel_w/2 + rim_inset);
}

module right_wall() {
    side_wall_wedge(panel_w/2 - rim_inset - rim_t);
}

// =========================
// UPPER ANGLED LIP
// =========================
//
// 1 mm inset from the left/right edges,
// extends backward and rises slightly.
//

module top_angled_lip() {
    span_x = panel_w - 2*rim_inset;
    eps = 0.02;

    hull() {
        // front strip at top inner edge
        translate([0, y_lip_front_center, plate_t + eps/2])
            cube([span_x, top_lip_t, eps], center = true);

        // rear strip, slightly higher
        translate([0, y_lip_back_center, plate_t + top_lip_depth - eps/2])
            cube([span_x, top_lip_t, eps], center = true);
    }
}

// =========================
// PANEL ASSEMBLY
// =========================

module panel() {
    union() {
        front_plate();
        left_wall();
        right_wall();
        top_angled_lip();
    }
}

// =========================
// PREVIEW
// =========================

module dummy_screen() {
    color([0.1, 0.1, 0.1, 0.45])
        translate([screen_cx, screen_cy, plate_t])
            linear_extrude(height = dummy_screen_t, convexity = 10)
                rounded_rect_2d(screen_w - 1.0, screen_h - 1.0, 1.5);
}

module dummy_screen_keepout() {
    if (show_keepout_preview) {
        color([0.8, 0.2, 0.2, 0.20])
            translate([screen_cx, screen_cy, plate_t + 0.2])
                linear_extrude(height = 8.0, convexity = 10)
                    rounded_rect_2d(screen_keepout_w, screen_keepout_h, 2.0);
    }
}

module dummy_camera() {
    color([0.2, 0.5, 0.8, 0.45])
        translate([camera_cx, camera_cy, plate_t])
            cylinder(h = 8.0, d = dummy_camera_d, $fn = 48);
}

module dummy_pir() {
    color([0.85, 0.85, 0.85, 0.5])
        translate([pir_cx, pir_cy, plate_t])
            cylinder(h = 10.0, d = dummy_pir_d, $fn = 72);
}

module dummy_buttons() {
    color([0.2, 0.2, 0.2, 0.5]) {
        translate([button1_cx, button_cy, plate_t])
            cylinder(h = 10.0, d = dummy_button_d, $fn = 48);

        translate([button2_cx, button_cy, plate_t])
            cylinder(h = 10.0, d = dummy_button_d, $fn = 48);
    }
}

module preview() {
    panel();
    dummy_screen();
    dummy_screen_keepout();
    dummy_camera();
    dummy_pir();
    dummy_buttons();
}

// =========================
// OUTPUT
// =========================

if (part == "preview") {
    preview();
} else {
    panel();
}