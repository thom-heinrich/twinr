//
// FS2108X integrated 3-part bearingless turntable
// supportless prototype with tab support
//
// Parts:
//   - top_wheel
//   - lower_ring
//   - servo_pod
//
// Uses ORIGINAL servo horn.
// No printed spline.
//
// Servo is aligned by OUTPUT AXIS, not by body center.
// Mounting tabs are supported on internal shoulders inside servo_pod.
//

part = "all";   // ["top_wheel","lower_ring","servo_pod","all","assembly_preview"]

$fn = 128;

// ================================================================
// Servo: FS2108X
// ================================================================

servo_body_x = 35.8;
servo_body_y = 15.0;
servo_body_h = 32.0;

// user-measured:
// distance from LEFT body side (cable side) to shaft center
servo_output_from_left_x = 10.0;

// total length incl. mounting tabs
servo_total_x_with_tabs = 49.0;

// FDM fit
servo_body_clearance_x = 0.35;
servo_body_clearance_y = 0.45;
servo_tab_clearance_x  = 0.35;

// side bulges / cylindrical protrusions on both sides
// approximated support-free by widening upward in Y
servo_side_boss_extra_each    = 1.7;
servo_side_relief_start_z     = 5.0;
servo_side_relief_transition_h = 4.0;

// ================================================================
// User-corrected vertical geometry
// ================================================================

// overall servo pod height
pod_h = 32.0;

// top surface of the support ledges for the servo tabs ("Flügel")
servo_tab_support_z = 21.5;

// ================================================================
// Global dimensions
// ================================================================

turntable_d   = 140;
lower_ring_od = 126;
lower_ring_id = 44;

top_plate_t   = 5.0;
lower_ring_t  = 4.0;

// ================================================================
// Bearingless ring geometry
// ================================================================

bearing_rib_outer_d = 110;
bearing_rib_width   = 10;
bearing_rib_h       = 2.2;

bearing_radial_clearance = 0.35;
bearing_axial_clearance  = 0.35;

bearing_groove_outer_d = bearing_rib_outer_d + 2*bearing_radial_clearance;
bearing_groove_inner_d = (bearing_rib_outer_d - 2*bearing_rib_width) - 2*bearing_radial_clearance;
bearing_groove_depth   = bearing_rib_h + bearing_axial_clearance;

// only the rib should carry
ring_top_relief_depth    = 0.8;
ring_relief_inner_margin = 1.0;
ring_relief_outer_margin = 1.0;

// ================================================================
// Horn interface on top wheel
// ================================================================

horn_boss_d = 28;
horn_boss_h = 2.2;

center_clear_d     = 40;
center_clear_depth = 3.2;

center_screw_access_d = 4.2;

// pilot holes for included horn
horn_pilot_d = 1.6;
horn_pilot_radii = [6, 9.5, 13];

// optional mounting holes on top
top_mount_hole_d     = 3.2;
top_mount_circle_d   = 56;
top_mount_hole_count = 6;

// ================================================================
// Servo pod / underbody
// ================================================================

pod_wall_t = 2.4;

// top flange / upper plate
pod_flange_t = 3.6;
pod_flange_overhang = 8.0;

// pegs between pod and lower ring
pod_peg_d = 4.8;
pod_peg_h = 3.0;
pod_peg_hole_clearance = 0.25;

// flange support margin around peg footprints
peg_pad_margin = 3.0;

// bottom frame so servo is supported
bottom_frame_t = 2.4;
bottom_open_x_margin = 8.0;
bottom_open_y_margin = 4.0;

// cable cutout on LEFT X side
pod_cable_slot_y = 11.0;
pod_cable_slot_h = 24.0;
pod_cable_slot_z0 = 0.0;

// additional bottom cable notch
bottom_cable_notch_y = 11.0;

// ================================================================
// Derived asymmetric geometry
// ================================================================

// body cavity extents relative to shaft center X=0
body_left_mag  = servo_output_from_left_x + servo_body_clearance_x;
body_right_mag = (servo_body_x - servo_output_from_left_x) + servo_body_clearance_x;

body_x_total  = body_left_mag + body_right_mag;
body_center_x = (body_right_mag - body_left_mag) / 2;

// tab cavity extents relative to shaft center X=0
tab_extra_each = (servo_total_x_with_tabs - servo_body_x) / 2;

tab_left_mag  = servo_output_from_left_x + tab_extra_each + servo_tab_clearance_x;
tab_right_mag = (servo_body_x - servo_output_from_left_x) + tab_extra_each + servo_tab_clearance_x;

tab_x_total  = tab_left_mag + tab_right_mag;
tab_center_x = (tab_right_mag - tab_left_mag) / 2;

// Y clearances
cavity_y_base = servo_body_y + 2*servo_body_clearance_y;
cavity_y_wide = cavity_y_base + 2*servo_side_boss_extra_each;

// outer sleeve around widest cavity
outer_left_mag  = tab_left_mag  + pod_wall_t;
outer_right_mag = tab_right_mag + pod_wall_t;

outer_x_total  = outer_left_mag + outer_right_mag;
outer_center_x = (outer_right_mag - outer_left_mag) / 2;

outer_y = cavity_y_wide + 2*pod_wall_t;

// peg positions
pod_peg_x_left  = outer_center_x - outer_x_total/2 - 3.8;
pod_peg_x_right = outer_center_x + outer_x_total/2 + 3.8;
pod_peg_y       = outer_y/2 + 6.0;

// top flange enlarged so peg footprints are fully supported
flange_left_mag_base  = outer_left_mag  + pod_flange_overhang;
flange_right_mag_base = outer_right_mag + pod_flange_overhang;
flange_y_base         = outer_y + 2*pod_flange_overhang;

flange_left_mag  = max(flange_left_mag_base,  -pod_peg_x_left  + pod_peg_d/2 + peg_pad_margin);
flange_right_mag = max(flange_right_mag_base,  pod_peg_x_right + pod_peg_d/2 + peg_pad_margin);
flange_y_half    = max(flange_y_base/2, pod_peg_y + pod_peg_d/2 + peg_pad_margin);

flange_x_total  = flange_left_mag + flange_right_mag;
flange_center_x = (flange_right_mag - flange_left_mag) / 2;
flange_y        = 2 * flange_y_half;

// bottom opening
bottom_open_x = body_x_total - bottom_open_x_margin;
bottom_open_y = cavity_y_base - bottom_open_y_margin;

// extents for layout
pod_left_mag  = max(flange_left_mag,  -pod_peg_x_left  + pod_peg_d/2);
pod_right_mag = max(flange_right_mag,  pod_peg_x_right + pod_peg_d/2);

// ================================================================
// Helpers
// ================================================================

module ring(od, id, h) {
    difference() {
        cylinder(d=od, h=h);
        translate([0,0,-0.1]) cylinder(d=id, h=h+0.2);
    }
}

module centered_cube(size_xyz) {
    translate([-size_xyz[0]/2, -size_xyz[1]/2, 0]) cube(size_xyz);
}

module centered_square(size_xy) {
    translate([-size_xy[0]/2, -size_xy[1]/2]) square(size_xy);
}

module polar_holes(n, circle_d, hole_d, h) {
    for (i = [0:n-1]) {
        rotate([0,0,i*360/n])
            translate([circle_d/2, 0, 0])
                cylinder(d=hole_d, h=h);
    }
}

module horn_pilot_matrix(h) {
    for (r = horn_pilot_radii) {
        for (a = [0:45:315]) {
            rotate([0,0,a])
                translate([r,0,0])
                    cylinder(d=horn_pilot_d, h=h);
        }
    }
}

module pod_peg_holes(h) {
    for (px = [pod_peg_x_left, pod_peg_x_right]) {
        for (sy = [-1,1]) {
            translate([px, sy*pod_peg_y, 0])
                cylinder(d=pod_peg_d + pod_peg_hole_clearance, h=h);
        }
    }
}

module pod_pegs(h) {
    for (px = [pod_peg_x_left, pod_peg_x_right]) {
        for (sy = [-1,1]) {
            translate([px, sy*pod_peg_y, 0])
                cylinder(d=pod_peg_d, h=h);
        }
    }
}

// ================================================================
// Part 1: upper rotating wheel
// raw orientation: underside at Z=0
// ================================================================

module top_wheel_raw() {
    difference() {
        union() {
            cylinder(d=turntable_d, h=top_plate_t);

            translate([0,0,-horn_boss_h])
                cylinder(d=horn_boss_d, h=horn_boss_h);
        }

        // bearing groove
        translate([0,0,-0.05])
            ring(
                bearing_groove_outer_d,
                bearing_groove_inner_d,
                bearing_groove_depth + 0.1
            );

        // center clearance
        translate([0,0,-0.05])
            cylinder(d=center_clear_d, h=center_clear_depth + 0.1);

        // center screw access
        translate([0,0,-horn_boss_h - 0.1])
            cylinder(d=center_screw_access_d, h=top_plate_t + horn_boss_h + 0.2);

        // horn pilot matrix
        translate([0,0,-horn_boss_h - 0.1])
            horn_pilot_matrix(top_plate_t + horn_boss_h + 0.2);

        // optional top mounting holes
        translate([0,0,-0.1])
            polar_holes(top_mount_hole_count, top_mount_circle_d, top_mount_hole_d, top_plate_t + 0.2);
    }
}

module top_wheel_print() {
    translate([0,0,top_plate_t])
        rotate([180,0,0])
            top_wheel_raw();
}

// ================================================================
// Part 2: lower bearing ring
// ================================================================

module lower_ring_part() {
    difference() {
        union() {
            ring(lower_ring_od, lower_ring_id, lower_ring_t);

            translate([0,0,lower_ring_t])
                ring(
                    bearing_rib_outer_d,
                    bearing_rib_outer_d - 2*bearing_rib_width,
                    bearing_rib_h
                );
        }

        // inner recessed top zone
        translate([0,0,lower_ring_t - ring_top_relief_depth])
            ring(
                bearing_groove_inner_d - 2*ring_relief_inner_margin,
                lower_ring_id,
                ring_top_relief_depth + 0.2
            );

        // outer recessed top zone
        translate([0,0,lower_ring_t - ring_top_relief_depth])
            ring(
                lower_ring_od - 2,
                bearing_groove_outer_d + 2*ring_relief_outer_margin,
                ring_top_relief_depth + 0.2
            );

        // peg holes for servo pod
        translate([0,0,-0.1])
            pod_peg_holes(lower_ring_t + 0.2);
    }
}

// ================================================================
// Part 3: separate servo pod with tab shoulders
// ================================================================

module servo_pod_part() {

    y_lower_straight_h = max(0.2, servo_side_relief_start_z - bottom_frame_t);
    y_upper_z = servo_side_relief_start_z + servo_side_relief_transition_h;
    y_scale = cavity_y_wide / cavity_y_base;

    difference() {
        union() {
            // outer sleeve
            difference() {
                translate([outer_center_x, 0, 0])
                    centered_cube([outer_x_total, outer_y, pod_h]);

                union() {
                    // lower body cavity, narrow in Y
                    translate([body_center_x, 0, bottom_frame_t])
                        linear_extrude(height=y_lower_straight_h)
                            centered_square([body_x_total, cavity_y_base]);

                    // support-free Y widening
                    translate([body_center_x, 0, servo_side_relief_start_z])
                        linear_extrude(height=servo_side_relief_transition_h, scale=[1, y_scale])
                            centered_square([body_x_total, cavity_y_base]);

                    // body cavity up to the support ledges
                    translate([body_center_x, 0, y_upper_z - 0.01])
                        linear_extrude(height=max(0.2, servo_tab_support_z - y_upper_z + 0.02))
                            centered_square([body_x_total, cavity_y_wide]);

                    // above the ledges: wider in X so the tabs have room
                    // this leaves internal shoulders at servo_tab_support_z
                    translate([tab_center_x, 0, servo_tab_support_z])
                        linear_extrude(height=pod_h - servo_tab_support_z + 0.2)
                            centered_square([tab_x_total, cavity_y_wide]);
                }
            }

            // enlarged top flange / upper plate
            translate([flange_center_x, 0, pod_h])
            difference() {
                centered_cube([flange_x_total, flange_y, pod_flange_t]);

                // opening clears the widened upper section incl. tabs
                translate([tab_center_x - flange_center_x, 0, -0.1])
                    centered_cube([tab_x_total + 0.8, cavity_y_wide + 0.8, pod_flange_t + 0.2]);
            }

            // pegs directly on the enlarged top flange
            translate([0,0,pod_h + pod_flange_t])
                pod_pegs(pod_peg_h);
        }

        // bottom opening aligned to lower body cavity
        translate([body_center_x, 0, -0.1])
            centered_cube([bottom_open_x, bottom_open_y, bottom_frame_t + 0.2]);

        // LEFT side cable slot through wall, from bottom upward
        translate([
            -outer_left_mag - 0.1,
            -pod_cable_slot_y/2,
            pod_cable_slot_z0
        ])
        cube([
            outer_left_mag - body_left_mag + 0.6,
            pod_cable_slot_y,
            pod_cable_slot_h
        ]);

        // extra bottom cable notch on LEFT side
        translate([
            -outer_left_mag - 0.1,
            -bottom_cable_notch_y/2,
            -0.1
        ])
        cube([
            outer_left_mag - body_left_mag + 0.6,
            bottom_cable_notch_y,
            bottom_frame_t + 0.2
        ]);
    }
}

// ================================================================
// Assembly preview
// ================================================================

module assembly_preview() {
    color([0.75,0.80,0.85,1.0])
        servo_pod_part();

    color([0.50,0.65,0.80,1.0])
        translate([0,0,pod_h + pod_flange_t])
            lower_ring_part();

    color([0.85,0.65,0.35,0.85])
        translate([0,0,pod_h + pod_flange_t + lower_ring_t])
            top_wheel_raw();
}

// ================================================================
// Layout
// ================================================================

layout_gap = 12;

top_half  = turntable_d / 2;
ring_half = lower_ring_od / 2;

top_x  = -(top_half + ring_half + layout_gap);
ring_x = 0;
pod_x  =  ring_half + layout_gap + pod_left_mag;

if (part == "top_wheel") {
    top_wheel_print();
}
else if (part == "lower_ring") {
    lower_ring_part();
}
else if (part == "servo_pod") {
    servo_pod_part();
}
else if (part == "assembly_preview") {
    assembly_preview();
}
else if (part == "all") {
    translate([top_x,  0, 0]) top_wheel_print();
    translate([ring_x, 0, 0]) lower_ring_part();
    translate([pod_x,  0, 0]) servo_pod_part();
}