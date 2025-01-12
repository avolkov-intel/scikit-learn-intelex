/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/csr.hpp"

#ifdef ONEDAL_DPCTL_INTEGRATION
#include "onedal/datatypes/data_conversion_dpctl.hpp"
#endif // ONEDAL_DPCTL_INTEGRATION

#include "onedal/datatypes/data_conversion.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

static void* init_numpy() {
    import_array();
    return nullptr;
}

ONEDAL_PY_INIT_MODULE(table) {
    init_numpy();

    py::class_<table> table_obj(m, "table");
    table_obj.def(py::init());
    table_obj.def_property_readonly("has_data", &table::has_data);
    table_obj.def_property_readonly("column_count", &table::get_column_count);
    table_obj.def_property_readonly("row_count", &table::get_row_count);
    table_obj.def_property_readonly("kind", [](const table& t) {
        if (t.get_kind() == 0) { // TODO: expose empty table kind
            return "empty";
        }
        if (t.get_kind() == homogen_table::kind()) {
            return "homogen";
        }
        if (t.get_kind() == detail::csr_table::kind()) {
            return "csr";
        }
        return "unknown";
    });

#ifdef ONEDAL_DPCTL_INTEGRATION
    define_sycl_usm_array_property(table_obj);
#endif // ONEDAL_DPCTL_INTEGRATION

    m.def("to_table", [](py::object obj) {
        auto* obj_ptr = obj.ptr();
        return convert_to_table(obj_ptr);
    });

    m.def("from_table", [](dal::table& t) -> py::handle {
        auto* obj_ptr = convert_to_pyobject(t);
        return obj_ptr;
    });

#ifdef ONEDAL_DPCTL_INTEGRATION
    m.def("dpctl_to_table", [](py::object obj) {
        return convert_from_dptensor(obj);
    });

#endif // ONEDAL_DPCTL_INTEGRATION
}

} // namespace oneapi::dal::python
