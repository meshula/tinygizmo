// This is free and unencumbered software released into the public domain.
// For more information, please refer to <http://unlicense.org>

#ifndef tinygizmo_hpp
#define tinygizmo_hpp

#include <functional> /// @TODO remove when the render callback is removed
#include <vector>     /// @TODO ditto
#include <stdint.h> // for uint32_t

namespace tinygizmo
{

    ///////////////
    //   Types   //
    ///////////////

    struct v2f { float x, y; };
    struct v3f { float x, y, z; };
    struct v4f { float x, y, z, w; };
    struct uint3 {
        uint32_t x, y, z;
        int operator[] (int j) const { return (&x)[j]; }
    };
    typedef v4f quatf;

    struct m44f {
        v4f x, y, z, w;
        constexpr const v4f& operator[] (int j) const { return (&x)[j]; }
        v4f& operator[] (int j) { return (&x)[j]; }
    };

    ///////////////
    //   Gizmo   //
    ///////////////

    struct rigid_transform
    {
        rigid_transform() {}
        rigid_transform(const v4f& orientation, const v3f& position, const v3f& scale) : orientation(orientation), position(position), scale(scale) {}
        rigid_transform(const v4f& orientation, const v3f& position, float scale) : orientation(orientation), position(position), scale{ scale, scale, scale } {}
        rigid_transform(const v4f& orientation, const v3f& position) : orientation(orientation), position(position) {}

        v3f      position{ 0,0,0 };
        v4f      orientation{ 0,0,0,1 };
        v3f      scale{ 1,1,1 };

        bool     uniform_scale() const { return scale.x == scale.y && scale.x == scale.z; }
        m44f     matrix() const;
        v3f      transform_vector(const v3f& vec) const;
        v3f      transform_point(const v3f& p) const;
        v3f      detransform_point(const v3f& p) const;
        v3f      detransform_vector(const v3f& vec) const;
    };


    /// @TODO remove these from here
    struct geometry_vertex { v3f position, normal; v4f color; };
    struct geometry_mesh { std::vector<geometry_vertex> vertices; std::vector<uint3> triangles; };

    enum class transform_mode
    {
        translate,
        rotate,
        scale
    };

    struct camera_parameters
    {
        float yfov, near_clip, far_clip;
        v3f position_;
        v4f orientation_;
    };

    struct gizmo_application_state
    {
        bool mouse_left{ false };
        bool hotkey_translate{ false };
        bool hotkey_rotate{ false };
        bool hotkey_scale{ false };
        bool hotkey_local{ false };
        bool hotkey_ctrl{ false };
        float screenspace_scale{ 0.f };     // If > 0.f, the gizmos are drawn scale-invariant with a screenspace value defined here
        float snap_translation{ 0.f };      // World-scale units used for snapping translation
        float snap_scale{ 0.f };            // World-scale units used for snapping scale
        float snap_rotation{ 0.f };         // Radians used for snapping rotation quaternions (i.e. PI/8 or PI/16)
        v2f viewport_size_;                 // 3d viewport used to render the view
        v3f ray_origin_;                    // world-space ray origin (i.e. the camera position)
        v3f ray_direction_;                 // world-space ray direction
        camera_parameters cam;              // Used for constructing inverse view projection for raycasting onto gizmo geometry
    };

    /// @TODO remove render callback
    class gizmo_context
    {
        struct gizmo_context_impl;
        gizmo_context_impl* impl;

    public:
        gizmo_context();
        ~gizmo_context();

        void update(const gizmo_application_state& state);         // Clear geometry buffer and update internal `gizmo_application_state` data
        void draw();                                                // Trigger a render callback per call to `update(...)`
        transform_mode get_mode() const;                            // Return the active mode being used by `transform_gizmo(...)`
        std::function<void(const geometry_mesh & r)> render;        // Callback to render the gizmo meshes

        // returns true if the gizmo is hovered, or being manipulated
        /// @TODO use char *const*;
        bool transform_gizmo(char const*const name, rigid_transform& t);
    };


} // end namespace tinygizmo;

#endif // end tinygizmo_hpp
