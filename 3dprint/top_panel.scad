//
// Top cover / Oberteil
// KORRIGIERT NACH SKIZZE:
//
// - Frontlippe = 20 mm horizontal
//   -> Innenwinkel zur 15°-Hauptflaeche = exakt 15°
// - beide Seitenwaende gleich ausgerichtet
//   -> keine "fliegende" Seite mehr
// - hinten echter senkrechter Abschluss (90° zum Boden)
// - Bodenreferenz ist z = 0
//
// Export:
//   part = "cover";   // STL
//   part = "preview"; // Vorschau
//

part = "cover";   // "cover" | "preview"

// =========================
// HAUPTPARAMETER
// =========================

cover_w = 200.0;

// Skizze
main_len            = 150.0;   // 15 cm
main_angle_deg      = 15.0;    // 15°
rear_len            = 40.0;    // 4 cm
peak_inner_angle_deg= 135.0;   // 135°
front_lip_len       = 20.0;    // 2 cm

top_t      = 2.0;

// Seitenwaende
wall_inset = 1.0;
wall_t     = 1.2;

// Hinten senkrechter Abschluss
rear_wall_t = 1.2;

// =========================
// CUTOUTS
// =========================

// grosses Cutout im 150-mm-Segment
main_cutout_w  = 90.0;
main_cutout_l  = 60.0;
main_cutout_cx = 0.0;
main_cutout_cy = main_len / 2;

// kleines Cutout im 40-mm-Segment
rear_cutout_w  = 40.0;   // 4 cm
rear_cutout_l  = 30.0;   // 3 cm
rear_cutout_cx = 0.0;
rear_cutout_cy = rear_len / 2;

// =========================
// ABGELEITETE WERTE
// =========================

// Ende Hauptsegment
main_end_y = main_len * cos(main_angle_deg);
main_end_z = main_len * sin(main_angle_deg);

// Hinteres Segment:
// 15° vorne + 135° Innenwinkel => hinten -30° gegen Horizontalen
rear_angle_deg = main_angle_deg + peak_inner_angle_deg - 180.0;

// Ende hinteres Segment
rear_end_y = main_end_y + rear_len * cos(rear_angle_deg);
rear_end_z = main_end_z + rear_len * sin(rear_angle_deg);

// Nur zur Kontrolle
front_lip_angle_deg   = 0.0;  // horizontal
front_inner_angle_deg = abs(main_angle_deg - front_lip_angle_deg);
rear_to_vertical_deg  = 90.0 - abs(rear_angle_deg);

// Innenbreite zwischen Insets
inner_w = cover_w - 2*wall_inset;

// X-Positionen der Seitenwaende
x_left_wall  = -cover_w/2 + wall_inset;
x_right_wall =  cover_w/2 - wall_inset - wall_t;

echo("main_end_y =", main_end_y);
echo("main_end_z =", main_end_z);
echo("rear_angle_deg =", rear_angle_deg);
echo("rear_end_y =", rear_end_y);
echo("rear_end_z =", rear_end_z);
echo("front_inner_angle_deg =", front_inner_angle_deg);
echo("rear_to_vertical_deg =", rear_to_vertical_deg);

assert(abs(front_inner_angle_deg - 15.0) < 0.0001,
    "Front-Innenwinkel ist nicht 15°.");
assert(rear_end_z > 0,
    "Hinteres Segment endet nicht ueber dem Boden.");
assert(front_lip_len > 0,
    "front_lip_len muss > 0 sein.");

// =========================
// HELFER
// =========================

module local_panel_2d(panel_w, panel_l, cut_w=0, cut_l=0, cut_cx=0, cut_cy=0) {
    difference() {
        translate([-panel_w/2, 0])
            square([panel_w, panel_l], center = false);

        if (cut_w > 0 && cut_l > 0)
            translate([cut_cx, cut_cy])
                square([cut_w, cut_l], center = true);
    }
}

// =========================
// OBERTEILSEGMENTE
// =========================

module main_segment() {
    rotate([main_angle_deg, 0, 0])
        linear_extrude(height = top_t, convexity = 10)
            local_panel_2d(
                cover_w,
                main_len,
                main_cutout_w,
                main_cutout_l,
                main_cutout_cx,
                main_cutout_cy
            );
}

module rear_segment() {
    translate([0, main_end_y, main_end_z])
        rotate([rear_angle_deg, 0, 0])
            linear_extrude(height = top_t, convexity = 10)
                local_panel_2d(
                    cover_w,
                    rear_len,
                    rear_cutout_w,
                    rear_cutout_l,
                    rear_cutout_cx,
                    rear_cutout_cy
                );
}

// =========================
// FRONTLIPPE
// =========================
//
// Exakt die 2-cm-Lippe vorne, horizontal.
// Dadurch ist der Innenwinkel zur 15°-Hauptflaeche exakt 15°.
//

module front_lip_segment() {
    translate([
        -cover_w/2 + wall_inset,
        0,
        0
    ])
        cube([
            inner_w,
            front_lip_len,
            top_t
        ], center = false);
}

// =========================
// HINTERER SENKRECHTER ABSCHLUSS
// =========================
//
// 90° zum Boden (z=0).
//

module rear_end_wall() {
    translate([
        -cover_w/2 + wall_inset,
        rear_end_y - rear_wall_t,
        0
    ])
        cube([
            inner_w,
            rear_wall_t,
            rear_end_z + top_t
        ], center = false);
}

// =========================
// SEITENWÄNDE
// =========================
//
// Seitenprofil jetzt passend zur Skizze:
// vordere Spitze -> Peak -> hinteres Segmentende
// -> senkrecht runter auf Boden
// -> vorne 20-mm-Lippe
//
// Kein Mirror mehr.
// Beide Seiten nutzen dieselbe Matrix-Transformation.
//

module side_wall_profile_2d() {
    polygon(points = [
        [0,            0],           // vordere Spitze
        [main_end_y,   main_end_z],  // Ende 15-cm-Segment
        [rear_end_y,   rear_end_z],  // Ende 4-cm-Segment
        [rear_end_y,   0],           // hinten senkrecht auf Boden
        [front_lip_len,0]            // untere Frontlippe 20 mm
    ]);
}

module side_wall_at(xoff) {
    multmatrix(m = [
        [0, 0, 1, xoff],  // local z -> global x
        [1, 0, 0, 0   ],  // local x -> global y
        [0, 1, 0, 0   ],  // local y -> global z
        [0, 0, 0, 1   ]
    ])
        linear_extrude(height = wall_t, convexity = 10)
            side_wall_profile_2d();
}

module left_wall() {
    side_wall_at(x_left_wall);
}

module right_wall() {
    side_wall_at(x_right_wall);
}

// =========================
// ASSEMBLY
// =========================

module cover() {
    union() {
        main_segment();
        rear_segment();

        front_lip_segment();
        rear_end_wall();

        left_wall();
        right_wall();
    }
}

// =========================
// PREVIEW
// =========================

module preview_axes() {
    color([1,0,0]) cube([40,1,1], center=false); // X
    color([0,1,0]) cube([1,40,1], center=false); // Y
    color([0,0,1]) cube([1,1,40], center=false); // Z
}

module preview() {
    cover();
    preview_axes();
}

// =========================
// OUTPUT
// =========================

if (part == "preview") {
    preview();
} else {
    cover();
}